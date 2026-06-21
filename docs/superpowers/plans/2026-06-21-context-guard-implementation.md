# Context-Window Guard Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Prevent `ochat.py` chat turns from getting their reply silently cut off by Ollama's context-length cap, and recover automatically (with a visible warning) on the rare turn where it still happens.

**Architecture:** Three additive changes to `ochat.py`, no new files. (1) A pure-logic function computes how much message history to send each turn based on how big that turn's system prompt actually is, instead of a fixed guess. (2) `ollama_chat` raises a new exception when Ollama reports the reply was cut off (`done_reason == "length"`) instead of silently returning the partial text. (3) `handle_turn` uses (1) for its budget and catches (2), retrying once with a smaller window before falling back to whatever partial text came back.

**Tech Stack:** Python 3.11+, `requests`, `numpy`, pytest (via `uv run --with pytest --with numpy --with requests`). No new dependencies.

## Global Constraints

- No new files — all changes live in `ochat.py` and `tests/test_memory.py`. (Spec: Architecture)
- No new dependencies.
- No automatic summarization/compaction of thread history. (Spec: Non-goals)
- No retry beyond one attempt — a second truncation gives up and saves the partial reply. (Spec: Non-goals)
- No changes to `extract_facts` or `classify_calendar_intent` — they already catch broad `Exception` and need no modification. (Spec: Non-goals)
- `RESPONSE_TOKEN_RESERVE = 2048`, used against the existing `OLLAMA_NUM_CTX = 16384` and `CONTEXT_TOKEN_BUDGET = 8192`. (Spec: Prevention)
- Follow this repo's existing TDD/mocking conventions exactly: pure-logic functions tested with plain values (no mocking); `requests`/`ollama_chat` calls mocked with `unittest.mock.patch`; `tmp_path` fixture for any filesystem-touching test. (CLAUDE.md: Testing conventions)
- Every new warning printed to the user follows the existing convention in `handle_turn`/`refresh_calendar_cache`: a leading `\n`, printed to `sys.stderr`.

---

### Task 1: Dynamic history budget (`effective_history_budget`)

**Files:**
- Modify: `ochat.py:31-32` (add constant), `ochat.py:73-83` (add function after `truncate_messages_to_budget`)
- Test: `tests/test_memory.py:80-84` (add tests after `test_truncate_messages_to_budget_always_keeps_newest_message`)

**Interfaces:**
- Produces: `ochat.RESPONSE_TOKEN_RESERVE` (int, `2048`); `ochat.effective_history_budget(system_prompt: str, num_ctx=OLLAMA_NUM_CTX, response_reserve=RESPONSE_TOKEN_RESERVE, max_budget=CONTEXT_TOKEN_BUDGET) -> int`. Task 3 calls this with just `system_prompt` (all defaults).

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_memory.py`, immediately after `test_truncate_messages_to_budget_always_keeps_newest_message` (currently ending at line 83):

```python
def test_effective_history_budget_uses_max_budget_when_system_prompt_small():
    result = ochat.effective_history_budget("short system prompt", num_ctx=16384, response_reserve=2048, max_budget=8192)
    assert result == 8192


def test_effective_history_budget_shrinks_for_large_system_prompt():
    huge_system_prompt = "x" * (7000 * 4)  # ~7000 estimated tokens
    result = ochat.effective_history_budget(huge_system_prompt, num_ctx=16384, response_reserve=2048, max_budget=8192)
    assert result == 16384 - 7000 - 2048
    assert result < 8192


def test_effective_history_budget_never_negative():
    enormous_system_prompt = "x" * (50000 * 4)  # ~50000 estimated tokens, far over num_ctx
    result = ochat.effective_history_budget(enormous_system_prompt, num_ctx=16384, response_reserve=2048, max_budget=8192)
    assert result == 0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd "/Volumes/JRAX82TB 1/PROJECTS/persememory" && uv run --with pytest --with numpy --with requests pytest tests/test_memory.py -v -k test_effective_history_budget`

Expected: 3 FAILED, each with `AttributeError: module 'ochat' has no attribute 'effective_history_budget'`.

- [ ] **Step 3: Implement the constant and function**

In `ochat.py`, change:

```python
CONTEXT_TOKEN_BUDGET = 8192
OLLAMA_NUM_CTX = 16384
CHARS_PER_TOKEN = 4
```

to:

```python
CONTEXT_TOKEN_BUDGET = 8192
OLLAMA_NUM_CTX = 16384
RESPONSE_TOKEN_RESERVE = 2048
CHARS_PER_TOKEN = 4
```

Then, in `ochat.py`, change:

```python
def truncate_messages_to_budget(messages, budget_tokens=CONTEXT_TOKEN_BUDGET):
    selected = []
    used = 0
    for message in reversed(messages):
        cost = estimate_tokens(message["content"])
        if selected and used + cost > budget_tokens:
            break
        selected.append(message)
        used += cost
    selected.reverse()
    return selected
```

to:

```python
def truncate_messages_to_budget(messages, budget_tokens=CONTEXT_TOKEN_BUDGET):
    selected = []
    used = 0
    for message in reversed(messages):
        cost = estimate_tokens(message["content"])
        if selected and used + cost > budget_tokens:
            break
        selected.append(message)
        used += cost
    selected.reverse()
    return selected


def effective_history_budget(system_prompt, num_ctx=OLLAMA_NUM_CTX,
                              response_reserve=RESPONSE_TOKEN_RESERVE,
                              max_budget=CONTEXT_TOKEN_BUDGET):
    available = num_ctx - estimate_tokens(system_prompt) - response_reserve
    return max(0, min(max_budget, available))
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd "/Volumes/JRAX82TB 1/PROJECTS/persememory" && uv run --with pytest --with numpy --with requests pytest tests/test_memory.py -v -k test_effective_history_budget`

Expected: 3 passed.

- [ ] **Step 5: Run the full suite to check for regressions**

Run: `cd "/Volumes/JRAX82TB 1/PROJECTS/persememory" && uv run --with pytest --with numpy --with requests pytest tests/ -v`

Expected: all tests pass (92 total: 89 existing + 3 new).

- [ ] **Step 6: Commit**

```bash
cd "/Volumes/JRAX82TB 1/PROJECTS/persememory" && git add ochat.py tests/test_memory.py && git commit -m "$(cat <<'EOF'
feat: add dynamic history budget based on system prompt size

effective_history_budget shrinks the per-turn history window when that
turn's system prompt (facts/calendar bullets) is large enough to otherwise
crowd out room for the model's reply within OLLAMA_NUM_CTX. Not yet wired
into handle_turn.
EOF
)"
```

---

### Task 2: `ResponseTruncatedError` — detect cut-off replies in `ollama_chat`

**Files:**
- Modify: `ochat.py:249-277` (`ollama_chat`, plus a new exception class defined just above it)
- Test: `tests/test_memory.py:1-7` (add `import pytest`), `tests/test_memory.py` (add test after `test_ollama_chat_sets_num_ctx_so_long_threads_dont_get_cut_off`)

**Interfaces:**
- Consumes: nothing new from Task 1.
- Produces: `ochat.ResponseTruncatedError` (Exception subclass with a `.text: str` attribute). Task 3 catches this exception type from `ochat.ollama_chat`.

- [ ] **Step 1: Write the failing test**

First, add `import pytest` to the top of `tests/test_memory.py` (after the existing `import sys`):

```python
import sys

import pytest
from pathlib import Path
```

Then add this test to `tests/test_memory.py`, immediately after `test_ollama_chat_sets_num_ctx_so_long_threads_dont_get_cut_off`:

```python
def test_ollama_chat_raises_response_truncated_error_when_done_reason_is_length():
    fake_response = MagicMock()
    fake_response.raise_for_status.return_value = None
    fake_response.iter_lines.return_value = [
        json.dumps({"message": {"content": "Hel"}, "done": False}).encode(),
        json.dumps({"message": {"content": "lo"}, "done": False}).encode(),
        json.dumps({"message": {"content": ""}, "done": True, "done_reason": "length"}).encode(),
    ]
    with patch("ochat.requests.post", return_value=fake_response):
        with pytest.raises(ochat.ResponseTruncatedError) as exc_info:
            ochat.ollama_chat([{"role": "user", "content": "hi"}], think=False, stream_to_stdout=False)
    assert exc_info.value.text == "Hello"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd "/Volumes/JRAX82TB 1/PROJECTS/persememory" && uv run --with pytest --with numpy --with requests pytest tests/test_memory.py -v -k test_ollama_chat_raises_response_truncated_error`

Expected: FAILED with `AttributeError: module 'ochat' has no attribute 'ResponseTruncatedError'`.

- [ ] **Step 3: Implement `ResponseTruncatedError` and update `ollama_chat`**

In `ochat.py`, change:

```python
def ollama_chat(messages, think=False, stream_to_stdout=True):
    response = requests.post(
        f"{OLLAMA_URL}/api/chat",
        json={
            "model": CHAT_MODEL,
            "messages": messages,
            "stream": True,
            "think": think,
            "options": {"num_ctx": OLLAMA_NUM_CTX},
        },
        stream=True,
        timeout=120,
    )
    response.raise_for_status()
    pieces = []
    for line in response.iter_lines():
        if not line:
            continue
        chunk = json.loads(line)
        piece = chunk.get("message", {}).get("content", "")
        if piece:
            pieces.append(piece)
            if stream_to_stdout:
                print(piece, end="", flush=True)
        if chunk.get("done"):
            break
    if stream_to_stdout:
        print()
    return "".join(pieces)
```

to:

```python
class ResponseTruncatedError(Exception):
    """Raised by ollama_chat when Ollama's done_reason is "length" -- the
    reply was cut off by the context/length cap rather than stopping
    naturally. Carries whatever partial text was generated."""

    def __init__(self, text: str):
        super().__init__("response was cut off: context window limit reached")
        self.text = text


def ollama_chat(messages, think=False, stream_to_stdout=True):
    response = requests.post(
        f"{OLLAMA_URL}/api/chat",
        json={
            "model": CHAT_MODEL,
            "messages": messages,
            "stream": True,
            "think": think,
            "options": {"num_ctx": OLLAMA_NUM_CTX},
        },
        stream=True,
        timeout=120,
    )
    response.raise_for_status()
    pieces = []
    done_reason = None
    for line in response.iter_lines():
        if not line:
            continue
        chunk = json.loads(line)
        piece = chunk.get("message", {}).get("content", "")
        if piece:
            pieces.append(piece)
            if stream_to_stdout:
                print(piece, end="", flush=True)
        if chunk.get("done"):
            done_reason = chunk.get("done_reason")
            break
    if stream_to_stdout:
        print()
    text = "".join(pieces)
    if done_reason == "length":
        raise ResponseTruncatedError(text)
    return text
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd "/Volumes/JRAX82TB 1/PROJECTS/persememory" && uv run --with pytest --with numpy --with requests pytest tests/test_memory.py -v -k test_ollama_chat_raises_response_truncated_error`

Expected: 1 passed.

- [ ] **Step 5: Run the full suite to check for regressions**

Run: `cd "/Volumes/JRAX82TB 1/PROJECTS/persememory" && uv run --with pytest --with numpy --with requests pytest tests/ -v`

Expected: all tests pass (93 total). In particular, `test_ollama_chat_streams_and_concatenates_content` must still pass unmodified — its mocked final chunk has no `done_reason` key, so `chunk.get("done_reason")` is `None`, which is `!= "length"`, so `ollama_chat` still returns normally.

- [ ] **Step 6: Commit**

```bash
cd "/Volumes/JRAX82TB 1/PROJECTS/persememory" && git add ochat.py tests/test_memory.py && git commit -m "$(cat <<'EOF'
feat: raise ResponseTruncatedError when Ollama cuts a reply off

ollama_chat now inspects done_reason on the final streamed chunk and raises
instead of silently returning truncated text when Ollama hit its context
cap (done_reason == "length"). Not yet handled by any caller.
EOF
)"
```

---

### Task 3: Wire the budget and retry into `handle_turn`, update docs

**Files:**
- Modify: `ochat.py:500-510` (the chat-call block inside `handle_turn`)
- Modify: `CLAUDE.md` (Configuration section + Notable hardening details section)
- Modify: `Persememory.md` (§8 Configuration table + §"Pure logic" table + §"Ollama client" table)
- Test: `tests/test_memory.py` (add tests after `test_handle_turn_saves_thread_and_starts_extraction_on_success`)

**Interfaces:**
- Consumes: `ochat.effective_history_budget(system_prompt)` from Task 1; `ochat.ResponseTruncatedError` (with `.text`) from Task 2.
- Produces: no new public interface — this is the integration point.

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_memory.py`, immediately after `test_handle_turn_saves_thread_and_starts_extraction_on_success`:

```python
def test_handle_turn_retries_once_with_smaller_window_after_truncated_reply(tmp_path, capsys):
    conn = ochat.init_db(tmp_path / "memory.db")
    thread = {"name": "t", "messages": []}
    path = tmp_path / "t.json"
    with patch("ochat.ollama_embed", return_value=np.array([1.0], dtype=np.float32)), \
         patch("ochat.ollama_chat", side_effect=[ochat.ResponseTruncatedError("partial"), "full reply"]) as mock_chat, \
         patch("ochat.extract_facts") as mock_extract:
        result = ochat.handle_turn(conn, thread, path, "hello", "off")
    assert result is not None
    result.join(timeout=2)
    assert mock_chat.call_count == 2
    assert thread["messages"][1]["content"] == "full reply"
    assert path.exists()
    assert "cut off" in capsys.readouterr().err
    mock_extract.assert_called_once()


def test_handle_turn_saves_partial_reply_when_retry_also_truncated(tmp_path, capsys):
    conn = ochat.init_db(tmp_path / "memory.db")
    thread = {"name": "t", "messages": []}
    path = tmp_path / "t.json"
    with patch("ochat.ollama_embed", return_value=np.array([1.0], dtype=np.float32)), \
         patch("ochat.ollama_chat", side_effect=[
             ochat.ResponseTruncatedError("first partial"),
             ochat.ResponseTruncatedError("second partial"),
         ]) as mock_chat, \
         patch("ochat.extract_facts") as mock_extract:
        result = ochat.handle_turn(conn, thread, path, "hello", "off")
    assert result is not None
    result.join(timeout=2)
    assert mock_chat.call_count == 2
    assert thread["messages"][1]["content"] == "second partial"
    assert path.exists()
    assert capsys.readouterr().err.count("cut off") == 2
    mock_extract.assert_called_once()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd "/Volumes/JRAX82TB 1/PROJECTS/persememory" && uv run --with pytest --with numpy --with requests pytest tests/test_memory.py -v -k "test_handle_turn_retries_once or test_handle_turn_saves_partial_reply"`

Expected: both FAILED. The first attempt's `ResponseTruncatedError` propagates straight out of `handle_turn` uncaught (today's code has no `except ResponseTruncatedError` handling), so both tests fail with `ochat.ResponseTruncatedError` raised instead of returning normally.

- [ ] **Step 3: Implement the retry in `handle_turn`**

In `ochat.py`, change:

```python
    try:
        system_prompt = build_system_prompt(relevant, calendar_events)
        window = truncate_messages_to_budget(
            thread["messages"] + [{"role": "user", "content": user_input}]
        )
        payload = [{"role": "system", "content": system_prompt}] + window
        print("ochat> ", end="", flush=True)
        reply = ollama_chat(payload, think=think_param(think))
    except requests.RequestException as exc:
        print(f"\nerror: chat request failed ({exc}); message not saved, try again", file=sys.stderr)
        return None
```

to:

```python
    try:
        system_prompt = build_system_prompt(relevant, calendar_events)
        budget = effective_history_budget(system_prompt)
        window = truncate_messages_to_budget(
            thread["messages"] + [{"role": "user", "content": user_input}], budget
        )
        payload = [{"role": "system", "content": system_prompt}] + window
        print("ochat> ", end="", flush=True)
        try:
            reply = ollama_chat(payload, think=think_param(think))
        except ResponseTruncatedError:
            print(
                "\nwarning: response was cut off (context limit reached); "
                "retrying with less history...",
                file=sys.stderr,
            )
            retry_window = truncate_messages_to_budget(
                thread["messages"] + [{"role": "user", "content": user_input}], budget // 2
            )
            retry_payload = [{"role": "system", "content": system_prompt}] + retry_window
            print("ochat> ", end="", flush=True)
            try:
                reply = ollama_chat(retry_payload, think=think_param(think))
            except ResponseTruncatedError as exc:
                print(
                    "\nwarning: response was still cut off after retrying with "
                    "less history; saving the partial reply",
                    file=sys.stderr,
                )
                reply = exc.text
    except requests.RequestException as exc:
        print(f"\nerror: chat request failed ({exc}); message not saved, try again", file=sys.stderr)
        return None
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd "/Volumes/JRAX82TB 1/PROJECTS/persememory" && uv run --with pytest --with numpy --with requests pytest tests/test_memory.py -v -k "test_handle_turn_retries_once or test_handle_turn_saves_partial_reply"`

Expected: both passed.

- [ ] **Step 5: Run the full suite to check for regressions**

Run: `cd "/Volumes/JRAX82TB 1/PROJECTS/persememory" && uv run --with pytest --with numpy --with requests pytest tests/ -v`

Expected: all tests pass (95 total: 93 from Tasks 1-2 + 2 new).

- [ ] **Step 6: Update `CLAUDE.md`**

In `CLAUDE.md`, change:

```
All tunables are plain module-level constants at the top of `ochat.py` —
no config file. Alongside the original set (`OLLAMA_URL`, `CHAT_MODEL`,
`EMBED_MODEL`, `CONTEXT_TOKEN_BUDGET`, `OLLAMA_NUM_CTX`, `CHARS_PER_TOKEN`, `RETRIEVAL_TOP_K`,
```

to:

```
All tunables are plain module-level constants at the top of `ochat.py` —
no config file. Alongside the original set (`OLLAMA_URL`, `CHAT_MODEL`,
`EMBED_MODEL`, `CONTEXT_TOKEN_BUDGET`, `OLLAMA_NUM_CTX`,
`RESPONSE_TOKEN_RESERVE`, `CHARS_PER_TOKEN`, `RETRIEVAL_TOP_K`,
```

Then, in `CLAUDE.md`, change:

```
- `ollama_chat` always sends `options: {"num_ctx": OLLAMA_NUM_CTX}`. Without
  it, Ollama loads the model at its own default context window (observed:
  4096 for `gemma4:12b`, far below the model's actual supported
  `context_length`), and since `num_ctx` caps prompt+completion *combined*,
  a long thread's history plus system prompt can crowd out nearly all of it,
  leaving too few tokens for the response and forcing Ollama to cut it off
  mid-sentence (`done_reason: "length"`). This was a real bug found via a
  live thread that had grown to 44 messages — both `OLLAMA_NUM_CTX` and
  `CONTEXT_TOKEN_BUDGET` need to stay configured (`OLLAMA_NUM_CTX` higher),
  since `CONTEXT_TOKEN_BUDGET` only bounds the sliding-window history, not
  the system prompt or the room the model needs to finish responding.
```

to:

```
- `ollama_chat` always sends `options: {"num_ctx": OLLAMA_NUM_CTX}`. Without
  it, Ollama loads the model at its own default context window (observed:
  4096 for `gemma4:12b`, far below the model's actual supported
  `context_length`), and since `num_ctx` caps prompt+completion *combined*,
  a long thread's history plus system prompt can crowd out nearly all of it,
  leaving too few tokens for the response and forcing Ollama to cut it off
  mid-sentence (`done_reason: "length"`). This was a real bug found via a
  live thread that had grown to 44 messages — both `OLLAMA_NUM_CTX` and
  `CONTEXT_TOKEN_BUDGET` need to stay configured (`OLLAMA_NUM_CTX` higher),
  since `CONTEXT_TOKEN_BUDGET` only bounds the sliding-window history, not
  the system prompt or the room the model needs to finish responding.
- `handle_turn` no longer hands `truncate_messages_to_budget` a flat
  `CONTEXT_TOKEN_BUDGET` — it calls `effective_history_budget(system_prompt)`
  first, which shrinks the history window when that turn's system prompt is
  big enough that `CONTEXT_TOKEN_BUDGET` worth of history plus the system
  prompt plus `RESPONSE_TOKEN_RESERVE` would exceed `OLLAMA_NUM_CTX`. If
  `ollama_chat` still raises `ResponseTruncatedError` (Ollama's
  `done_reason` was `"length"`), `handle_turn` retries once with half the
  budget; if that also gets cut off, it gives up and saves the retry's
  partial text rather than dropping the turn, printing a warning either way.
```

- [ ] **Step 7: Update `Persememory.md`**

In `Persememory.md`, change the `truncate_messages_to_budget` row in the "Pure logic (no I/O)" table:

```
| `truncate_messages_to_budget` | `(messages, budget_tokens=8192) -> list[dict]` | Walks messages newest-to-oldest, keeping as many as fit the budget; always keeps at least the single newest message even if it alone exceeds the budget |
```

to:

```
| `truncate_messages_to_budget` | `(messages, budget_tokens=8192) -> list[dict]` | Walks messages newest-to-oldest, keeping as many as fit the budget; always keeps at least the single newest message even if it alone exceeds the budget |
| `effective_history_budget` | `(system_prompt, num_ctx=16384, response_reserve=2048, max_budget=8192) -> int` | The actual `budget_tokens` `handle_turn` passes to `truncate_messages_to_budget`. Normally just `max_budget`; shrinks when this turn's `system_prompt` is large enough that `max_budget` of history plus the system prompt plus `response_reserve` would exceed `num_ctx`. Floored at `0`, never negative |
```

Then, in `Persememory.md`, change the `ollama_chat` row in the "Ollama client" table:

```
| `ollama_chat(messages, think, stream_to_stdout)` | Calls `/api/chat` with `stream: true` and `options: {"num_ctx": OLLAMA_NUM_CTX}`, parses the NDJSON response line-by-line, concatenates content chunks, and stops as soon as a chunk's `done` flag is set (verified to ignore any further lines after `done`). The explicit `num_ctx` matters: without it Ollama loads the model at its own default context window (observed: 4096 for `gemma4:12b`, independent of the model's much larger supported `context_length`), and once a thread's history plus system prompt approaches that ceiling, `num_ctx` caps prompt+completion *combined* — leaving too few tokens for the response and forcing Ollama to cut it off mid-sentence (`done_reason: "length"`) instead of stopping naturally. `OLLAMA_NUM_CTX` is set well above `CONTEXT_TOKEN_BUDGET` for exactly this reason: that budget only bounds the sliding-window *history*, not the system prompt (facts/calendar bullets) or the room the model needs to actually finish talking |
```

to:

```
| `ollama_chat(messages, think, stream_to_stdout)` | Calls `/api/chat` with `stream: true` and `options: {"num_ctx": OLLAMA_NUM_CTX}`, parses the NDJSON response line-by-line, concatenates content chunks, and stops as soon as a chunk's `done` flag is set (verified to ignore any further lines after `done`). The explicit `num_ctx` matters: without it Ollama loads the model at its own default context window (observed: 4096 for `gemma4:12b`, independent of the model's much larger supported `context_length`), and once a thread's history plus system prompt approaches that ceiling, `num_ctx` caps prompt+completion *combined* — leaving too few tokens for the response and forcing Ollama to cut it off mid-sentence (`done_reason: "length"`) instead of stopping naturally. `OLLAMA_NUM_CTX` is set well above `CONTEXT_TOKEN_BUDGET` for exactly this reason: that budget only bounds the sliding-window *history*, not the system prompt (facts/calendar bullets) or the room the model needs to actually finish talking. If the final chunk's `done_reason` is `"length"`, `ollama_chat` raises `ResponseTruncatedError(text)` (carrying the partial text via `.text`) instead of returning it — see `handle_turn` below for how this is recovered from |
| `ResponseTruncatedError` | Exception raised by `ollama_chat` when `done_reason == "length"`. `extract_facts`/`classify_calendar_intent` need no special handling — both already catch broad `Exception` around their `ollama_chat` calls and fall back safely |
```

Then, in `Persememory.md` §8 Configuration table, change:

```
| `CONTEXT_TOKEN_BUDGET` | `8192` | Approximate token budget for the sliding-window history sent per turn |
| `OLLAMA_NUM_CTX` | `16384` | The actual Ollama context window requested via `options.num_ctx` on every `ollama_chat` call. Kept above `CONTEXT_TOKEN_BUDGET` so the system prompt and the model's response still have room once the windowed history fills its budget; without this, Ollama silently falls back to its own default context window (4096 for `gemma4:12b`), which is smaller than `CONTEXT_TOKEN_BUDGET` itself and causes long-running threads to get cut off mid-response |
```

to:

```
| `CONTEXT_TOKEN_BUDGET` | `8192` | The maximum token budget for the sliding-window history sent per turn — an upper bound, not a guarantee; `effective_history_budget` may shrink it further on any given turn |
| `OLLAMA_NUM_CTX` | `16384` | The actual Ollama context window requested via `options.num_ctx` on every `ollama_chat` call. Kept above `CONTEXT_TOKEN_BUDGET` so the system prompt and the model's response still have room once the windowed history fills its budget; without this, Ollama silently falls back to its own default context window (4096 for `gemma4:12b`), which is smaller than `CONTEXT_TOKEN_BUDGET` itself and causes long-running threads to get cut off mid-response |
| `RESPONSE_TOKEN_RESERVE` | `2048` | Tokens reserved for the model's reply within `OLLAMA_NUM_CTX`, subtracted (along with the system prompt's size) when `effective_history_budget` computes that turn's actual history budget |
```

- [ ] **Step 8: Commit**

```bash
cd "/Volumes/JRAX82TB 1/PROJECTS/persememory" && git add ochat.py tests/test_memory.py CLAUDE.md Persememory.md && git commit -m "$(cat <<'EOF'
feat: guard chat turns against context-window truncation

handle_turn now sizes its history window with effective_history_budget
(Task 1) instead of a flat CONTEXT_TOKEN_BUDGET, and catches
ResponseTruncatedError (Task 2) to retry once with half the budget before
falling back to whatever partial reply came back. Closes the underlying
class of bug behind the recent OLLAMA_NUM_CTX fix, not just that one
incident.
EOF
)"
```

---

## Plan Self-Review Notes

- **Spec coverage:** Prevention (§"Prevention: dynamic history budget") → Task 1. Detection (§"Detection: ResponseTruncatedError") → Task 2. Recovery (§"Recovery: retry once") → Task 3. Error-handling table rows → covered by Task 3's tests (truncate-once-succeed, truncate-twice-fall-back) plus the pre-existing `test_handle_turn_returns_none_and_does_not_save_on_request_failure` (genuine HTTP failure path, untouched). Testing strategy section → each bullet maps 1:1 to a task's test step.
- **Type consistency:** `effective_history_budget(system_prompt, num_ctx, response_reserve, max_budget) -> int` (Task 1) is called in Task 3 as `effective_history_budget(system_prompt)` — matches, since Task 3 relies on the defaults bound to the same module-level constants. `ResponseTruncatedError(text)` (Task 2) exposes `.text` — Task 3 reads `exc.text` — matches.
- **No placeholders:** every step has literal before/after code or an exact shell command with expected output.
