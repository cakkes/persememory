# ochat — context-window guard

Date: 2026-06-21
Status: Approved for implementation

## Problem

A live bug was just root-caused and fixed: `ollama_chat` never told Ollama
what context window (`num_ctx`) to actually load the model with, so Ollama
fell back to its own server default (observed: 4096 tokens for
`gemma4:12b`) regardless of `ochat.py`'s own `CONTEXT_TOKEN_BUDGET` (8192)
bookkeeping constant. Once a thread's history grew large enough, the prompt
alone approached that 4096-token ceiling, leaving too little room for the
model to generate — Ollama cut the reply off mid-sentence
(`done_reason: "length"`) instead of stopping naturally, and `ochat.py`
saved the truncated reply with no warning. The immediate fix was adding
`OLLAMA_NUM_CTX = 16384` and passing `options.num_ctx` on every
`ollama_chat` call.

That fix closes the specific incident, but not the underlying class of bug:
`CONTEXT_TOKEN_BUDGET` only bounds the sliding-window *history* that
`ochat.py` sends — it has no idea how big that turn's system prompt
(instructions + up to 8 fact bullets + calendar event bullets) will end up
being, and nothing checks whether a reply actually got cut off after the
fact. A turn with an unusually large system prompt (e.g. a busy calendar
day) could still crowd out the model's response room, silently, even with
the larger `OLLAMA_NUM_CTX`.

## Goals

- Prevent the combined prompt (system prompt + windowed history) from
  crowding out room for the model's reply, adapting automatically to
  however large that turn's system prompt actually is — not a fixed guess.
- Detect the rare case where a reply still gets cut off despite the above,
  and recover automatically rather than silently saving a truncated reply.
- Keep ordinary turns (small system prompt, short threads) byte-for-byte
  unaffected — this is a guard for an edge case, not a behavior change for
  the common path.
- Stay within the existing single-file, no-config-file, plain-constants
  architecture described in `CLAUDE.md` — new logic lives in `ochat.py`'s
  existing "pure logic" and "Ollama client" layers, not a new file.

## Non-goals

- No automatic summarization/compaction of long thread history — out of
  scope; the guard only adjusts how much of the existing history is sent
  per turn, never rewrites or drops it from the saved thread file.
- No retry beyond one attempt — a second truncation gives up and saves the
  partial reply rather than looping.
- No change to `extract_facts` or `classify_calendar_intent` call sites —
  they already wrap `ollama_chat` in a broad `except Exception` with safe
  fallbacks, so the new exception (below) is handled by existing code
  without modification.

## Architecture

All changes are additions to `ochat.py`. No new files, no new dependencies.

```
ochat.py
  pure logic:      + RESPONSE_TOKEN_RESERVE (constant)
                    + effective_history_budget(...)
  Ollama client:    + ResponseTruncatedError
                    ~ ollama_chat(...)            (raises on done_reason == "length")
  orchestration:    ~ handle_turn(...)            (uses dynamic budget; retries once on truncation)
```

## Prevention: dynamic history budget

New constant, alongside `CONTEXT_TOKEN_BUDGET`:

```python
RESPONSE_TOKEN_RESERVE = 2048  # tokens reserved for the model's reply within OLLAMA_NUM_CTX
```

New pure-logic function, alongside `estimate_tokens`/`truncate_messages_to_budget`:

```python
def effective_history_budget(system_prompt, num_ctx=OLLAMA_NUM_CTX,
                              response_reserve=RESPONSE_TOKEN_RESERVE,
                              max_budget=CONTEXT_TOKEN_BUDGET) -> int:
    """How many tokens of message history to include this turn.

    Normally just max_budget (today's static CONTEXT_TOKEN_BUDGET). Shrinks
    below that only when this turn's system prompt is large enough that
    max_budget worth of history plus the system prompt plus the reserved
    response room would exceed num_ctx. Never negative.
    """
    available = num_ctx - estimate_tokens(system_prompt) - response_reserve
    return max(0, min(max_budget, available))
```

`handle_turn` builds `system_prompt` first (it already does, via
`build_system_prompt`), computes `budget = effective_history_budget(system_prompt)`,
and passes `budget` as the `budget_tokens` argument to
`truncate_messages_to_budget`, instead of relying on that function's
`CONTEXT_TOKEN_BUDGET` default. This `budget` local is what the retry path
below halves.

With `OLLAMA_NUM_CTX = 16384`, `RESPONSE_TOKEN_RESERVE = 2048`, and
`CONTEXT_TOKEN_BUDGET = 8192`: the budget only starts shrinking once a
single turn's system prompt exceeds roughly 6144 estimated tokens (~24KB of
fact/calendar text) — well above normal usage, so typical turns see no
change at all.

## Detection: `ResponseTruncatedError`

```python
class ResponseTruncatedError(Exception):
    """Raised by ollama_chat when Ollama's done_reason is "length" — the
    reply was cut off by the context/length cap rather than stopping
    naturally. Carries whatever partial text was generated."""
    def __init__(self, text: str):
        super().__init__("response was cut off: context window limit reached")
        self.text = text
```

`ollama_chat` already reads the final NDJSON chunk (the one with
`"done": true`) to know when to stop streaming; that same chunk carries
`done_reason`. After the streaming loop, if `done_reason == "length"`,
raise `ResponseTruncatedError(text)` instead of returning `text` normally.
Any other `done_reason` (`"stop"`, or absent in existing tests/mocks)
behaves exactly as today — this is purely additive.

`extract_facts` and `classify_calendar_intent` need no changes: both
already catch broad `Exception` around their `ollama_chat` calls and fall
back safely (`{"intent": "none", ...}` / skip fact insertion). A truncated
reply there now fails fast with a clear message instead of trying — and
likely failing — to JSON-parse cut-off output.

## Recovery: retry once in `handle_turn`

```python
try:
    reply = ollama_chat(payload, think=think_param(think))
except ResponseTruncatedError:
    print("warning: response was cut off (context limit reached); "
          "retrying with less history...", file=sys.stderr)
    retry_window = truncate_messages_to_budget(
        thread["messages"] + [{"role": "user", "content": user_input}],
        budget // 2,
    )
    retry_payload = [{"role": "system", "content": system_prompt}] + retry_window
    print("ochat> ", end="", flush=True)
    try:
        reply = ollama_chat(retry_payload, think=think_param(think))
    except ResponseTruncatedError as exc:
        print("warning: response was still cut off after retrying with "
              "less history; saving the partial reply", file=sys.stderr)
        reply = exc.text
```

This sits inside `handle_turn`'s existing call to `ollama_chat`, nested
inside the existing `try/except requests.RequestException` block (a
genuine HTTP failure still aborts the turn exactly as today —
`ResponseTruncatedError` is a distinct exception type and doesn't change
that path). The retry halves the just-used budget, trims more history, and
sends the same user input again. If the retry also gets cut off, the turn
isn't aborted — the partial text is saved (a degraded but real answer is
better than silently dropping the user's turn), with two distinct warnings
printed so the failure is visible.

**Known UX quirk, accepted as a tradeoff:** replies stream live to the
terminal as they arrive. The first (truncated) attempt's text is already
printed before `ollama_chat` finishes the stream and raises — by the time
the warning prints, the truncated text is already on screen, followed by
the warning, then the retry's `ochat> ` prefix and its own streamed text.
Buffering the first attempt to suppress this would mean not streaming
until a full reply completes, which contradicts this project's explicit
speed/responsiveness priority (see `Persememory.md` §1). This is expected
to be rare post-fix, so the tradeoff favors keeping live streaming intact.

## Error handling & resilience (additions to the existing table)

| Failure | Behavior |
|---|---|
| Turn's system prompt unusually large (busy calendar day, many facts) | `effective_history_budget` shrinks the history window for that turn only; no warning needed since no reply was actually cut off |
| Reply still cut off once (`done_reason == "length"`) | Warn, retry once with half the budget |
| Reply cut off again on retry | Warn again, save the partial reply rather than aborting the turn |
| Genuine HTTP failure (`requests.RequestException`) | Unchanged — turn aborts completely, nothing saved, exactly as today |

## Testing strategy

Following this repo's existing TDD/mocking conventions in
`tests/test_memory.py`:

- `effective_history_budget`: pure-value tests — normal case returns
  `max_budget` unchanged; a large `system_prompt` shrinks the result;
  an extreme `system_prompt` floors at `0`, never negative.
- `ollama_chat`: extend the existing mocked-response tests — a final chunk
  with `"done_reason": "length"` raises `ResponseTruncatedError` whose
  `.text` matches the concatenated streamed pieces; a final chunk with
  `"done_reason": "stop"` (or the key absent, as today's existing test
  already exercises) returns normally with no exception.
- `handle_turn`: mock `ochat.ollama_chat` with
  `side_effect=[ResponseTruncatedError("partial"), "full reply"]` and
  assert it's called twice, a warning is printed, and the thread ends up
  saving `"full reply"`; a second test with both calls raising
  `ResponseTruncatedError` asserts the thread saves the retry's `.text` and
  two warnings are printed.

## Follow-ups (out of scope for this version)

- No thread-history compaction/summarization — if a single thread keeps
  growing indefinitely, the sliding window still only shows the model the
  most recent slice; older content is preserved on disk but never
  re-surfaced. Not addressed here.
- No telemetry/counter for how often truncation happens — a stderr warning
  is considered sufficient for this single-user CLI tool.
