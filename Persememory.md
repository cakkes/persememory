# Persememory (ochat) — Full Documentation

## 1. Overview

`ochat` is a single-file Python CLI that replaces `ollama run gemma4:12b`
with a tool that gives a local Ollama-backed chat session two things plain
`ollama run` doesn't have:

1. **Conversation continuity** — closing the terminal and coming back later
   resumes the same thread, with full history.
2. **Long-term memory** — facts and preferences mentioned in any
   conversation are distilled and stored, and can be recalled by semantic
   search from a completely different conversation, even after the original
   exchange has scrolled out of the model's context window.

It was built to replace OpenClaw, an earlier general-purpose agent gateway
that was uninstalled for being too slow (hidden "thinking" tokens and
model-reload latency dominated response time). `ochat` is deliberately
minimal: no gateway process, no daemon, no plugins — it only runs while
you're actively using it, and every architectural choice below was made with
that speed complaint in mind.

### Goals

- Resume a named conversation thread across terminal sessions.
- Recall specific facts from arbitrarily old conversations via semantic
  search, since the long-term fact store is expected to grow large over
  time and a keyword search wouldn't scale to "find anything relevant."
- Stay fast: no background service, default "thinking" mode `off`.
- Stay simple and inspectable: plain JSON/SQLite files the user can read or
  edit directly, no config-file layer.

### Non-goals

- No multi-device sync.
- No GUI.
- No automatic pulling of models without explicit user action.
- No support for remote/non-Ollama model providers.

## 2. Architecture

```
~/ochat/
  ochat.py            # the entire tool — single file, ~440 lines
  tests/test_memory.py
```

`ochat.py` opens with a `uv run --script` shebang and a PEP 723 inline
dependency block:

```python
#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy", "requests"]
# ///
```

This means `uv` resolves and caches `numpy` + `requests` into an ephemeral
environment the first time the script runs — no `pip install`, no
`requirements.txt`, no committed virtualenv. The file is `chmod +x` and
symlinked onto `PATH` as `ochat`.

Despite being one file, the code is organized into four clear layers,
top to bottom in the source:

1. **Pure logic** — similarity scoring, dedup, token-budget truncation. No
   I/O, fully unit-testable with plain values.
2. **Storage** — thread JSON files and the SQLite fact store. Filesystem and
   database I/O, but no network calls.
3. **Ollama client** — HTTP calls to the local Ollama server (readiness
   check, embeddings, streaming chat).
4. **Orchestration + CLI** — wires the above into the interactive loop and
   the `threads`/`memory` subcommands.

Data lives entirely outside the repo, under `~/.local/share/ochat/`:

```
~/.local/share/ochat/
  threads/<name>.json   # one file per conversation thread, full history
  memory.db             # SQLite, long-term facts, shared across threads
  extraction.log        # background fact-extraction error log
```

## 3. Technology Stack

| Component | Choice | Why |
|---|---|---|
| Language/runtime | Python 3.11+ via `uv run --script` | No project-level dependency management; `uv` handles ephemeral deps from inline metadata |
| Chat model | `gemma4:12b` (Ollama) | Already running locally; Q4_K_M quantization, ~11.9B params, 100% GPU offload on the host's Apple M4 |
| Embedding model | `nomic-embed-text` (Ollama) | Small (~274MB), fast, dedicated solely to turning fact text and queries into vectors for similarity search |
| Vector search | Brute-force cosine similarity via `numpy` | No dedicated vector database — at personal-assistant scale (hundreds to low thousands of facts), a linear scan is fast enough and avoids another moving part |
| Conversation storage | Plain JSON files | Human-readable, trivially backed up, no schema migrations needed |
| Long-term memory storage | SQLite (stdlib `sqlite3`), WAL mode | Zero-install, single-file, safe for one writer (main thread) + one occasional background writer (extraction thread) |
| HTTP client | `requests` | Talks to Ollama's local REST API (`/api/version`, `/api/tags`, `/api/embeddings`, `/api/chat`) |
| Testing | `pytest`, run via `uv run --with pytest --with numpy --with requests` | Same ephemeral-dependency philosophy extended to the test run — no separate dev-dependency file |

## 4. Data Model

### Thread file — `threads/<name>.json`

```json
{
  "name": "work",
  "messages": [
    {"role": "user", "content": "...", "ts": "2026-06-19T07:09:35+00:00"},
    {"role": "assistant", "content": "...", "ts": "2026-06-19T07:09:41+00:00"}
  ]
}
```

The full history is kept forever — it is never pruned on disk. Only the
*payload sent to the model* on each turn is windowed (see §6).

### Long-term facts — `memory.db`

```sql
CREATE TABLE facts (
    id INTEGER PRIMARY KEY,
    text TEXT NOT NULL,
    embedding BLOB NOT NULL,     -- float32 vector from nomic-embed-text, raw bytes
    source_thread TEXT,
    created_at TEXT NOT NULL
);
```

`embedding` is stored as the raw bytes of a `float32` numpy array
(`np.asarray(embedding, dtype=np.float32).tobytes()`) and reconstituted on
read with `np.frombuffer(blob, dtype=np.float32)`. This keeps the table
schema-simple while avoiding any precision loss in the round trip.

## 5. Core Components (function reference)

### Pure logic (no I/O)

| Function | Signature | Purpose |
|---|---|---|
| `cosine_similarity` | `(a, b) -> float` | Standard cosine similarity; returns `0.0` for a zero-norm vector instead of dividing by zero |
| `top_k_facts` | `(query_embedding, facts, k=8, min_similarity=0.45) -> list[dict]` | Filters facts below the similarity cutoff, then returns the top `k` by score, highest first |
| `is_duplicate_fact` | `(candidate_embedding, existing_embeddings, threshold=0.92) -> bool` | True if the candidate is more than `threshold`-similar to anything already known |
| `estimate_tokens` | `(text) -> int` | `len(text) // 4`, floored at 1 — a cheap heuristic, not a real tokenizer |
| `truncate_messages_to_budget` | `(messages, budget_tokens=8192) -> list[dict]` | Walks messages newest-to-oldest, keeping as many as fit the budget; always keeps at least the single newest message even if it alone exceeds the budget |

### Storage

| Function | Purpose |
|---|---|
| `thread_path(name)` | Maps a thread name to its JSON file path |
| `load_thread(path, name)` | Loads a thread, or returns a fresh empty one if the file is missing. If the file exists but is corrupt/malformed, it's renamed aside (`<name>.json.corrupt-<timestamp>`) with a stderr warning, and a fresh thread is returned — the corrupt data is never deleted, just quarantined |
| `save_thread(path, thread)` | Atomic write: writes to a `.tmp` file, then `os.replace()`s it into place, so a crash mid-write can never leave a half-written thread file |
| `init_db(db_path)` | Opens (creating if needed) the SQLite database in WAL mode with `check_same_thread=False`, since the connection is shared between the main thread and background extraction threads |
| `insert_fact` / `get_all_facts` / `delete_fact` | Standard CRUD over the `facts` table |

### Ollama client

| Function | Purpose |
|---|---|
| `check_ollama_ready()` | Pings `/api/version`, then checks `/api/tags` for both required models. Exits the program with a clear, actionable message — never auto-pulls a model |
| `_model_installed(required, installed_names)` | Handles Ollama's default model tagging: a required name with no tag (`"nomic-embed-text"`) matches any tagged variant installed (`"nomic-embed-text:latest"`); a required name that already specifies a tag (`"gemma4:12b"`) only matches that exact tag |
| `ollama_embed(text)` | Calls `/api/embeddings` with the embedding model, returns a `float32` numpy vector |
| `ollama_chat(messages, think, stream_to_stdout)` | Calls `/api/chat` with `stream: true`, parses the NDJSON response line-by-line, concatenates content chunks, and stops as soon as a chunk's `done` flag is set (verified to ignore any further lines after `done`) |
| `think_param(level)` | Maps a CLI string (`"off"`, `"on"`/`"true"`, or a level name) to the value Ollama's `think` API field expects |

### Background fact extraction

| Function | Purpose |
|---|---|
| `EXTRACTION_PROMPT` | The fixed system prompt asking the model to return a JSON array of new durable facts from an exchange, or `[]` |
| `_extract_json_array(text)` | Strips markdown code fences (` ```json ... ``` `) and/or extracts the substring between the first `[` and last `]`, so prose-wrapped or fenced model output still parses as JSON |
| `log_extraction_error(exc)` | Appends a timestamped, `repr()`'d exception to `extraction.log` |
| `extract_facts(conn, user_message, assistant_message, source_thread)` | The whole extraction pipeline: calls the model, parses its JSON (via `_extract_json_array`), embeds and dedups each candidate fact (both against facts already in the database *and* against other facts extracted earlier in the same call), and inserts new ones. Wrapped in a blanket `except Exception` — this function is designed to **never** raise, since it always runs inside a background thread |

### Orchestration

| Function | Purpose |
|---|---|
| `build_system_prompt(relevant_facts)` | Builds the system prompt: a fixed base instruction, plus a `"Relevant memory:"` bullet list if any facts were retrieved |
| `handle_turn(conn, thread, path, user_input, think)` | Processes one full turn (see §6 for the detailed flow). Returns the background extraction `Thread` it started, or `None` if the turn failed before producing a reply |
| `run_chat_loop(thread_name, think)` | The interactive REPL: reads input, calls `handle_turn`, and on exit joins any still-running extraction thread with a bounded timeout so a quick quit doesn't usually lose the last exchange's facts |

### CLI

| Function | Purpose |
|---|---|
| `cmd_threads()` | Lists every thread file with message count and last-modified time |
| `cmd_memory_list()` | Prints every stored fact with its id, text, source thread, and creation time |
| `cmd_memory_forget(fact_id)` | Deletes a fact by id; exits with status 1 and a stderr message if the id doesn't exist |
| `main()` | `argparse`-based entrypoint wiring `--thread`, `--think`, `threads`, `memory list`, `memory forget <id>`, and the default (no subcommand) path into `run_chat_loop` |

## 6. Turn-by-Turn Flow

This is what happens for every message you send, traced through `handle_turn`:

1. **Embed the message.** `ollama_embed(user_input)` turns your message into
   a vector. If this fails (Ollama unreachable, model error), the turn aborts
   immediately: an error prints, nothing is written to the thread, and you
   can just resend the message.
2. **Retrieve relevant long-term facts.** `top_k_facts` scores your message's
   embedding against every fact in `memory.db` and keeps the best ≤8 matches
   above a 0.45 similarity cutoff. **This step is fault-isolated**: if the
   database read or scoring fails for any reason, the turn does *not* abort
   — it logs a warning and simply proceeds with an empty fact list, exactly
   as the original design called for ("SQLite errors during retrieval are
   caught; that turn proceeds with no relevant facts").
3. **Build the prompt.** The system prompt (base instructions + any
   retrieved facts) is combined with a sliding window of the most recent
   thread messages, trimmed by `truncate_messages_to_budget` to fit an
   8192-token estimated budget, plus your new message.
4. **Call the model.** `ollama_chat` streams the reply to your terminal as
   it's generated, with `think_param(think)` controlling reasoning depth
   (default `off`). If this network call fails, the turn aborts the same way
   step 1 does — no partial state is ever saved.
5. **Persist.** Both your message and the reply are appended to the
   in-memory thread and written to disk *atomically* via `save_thread`.
6. **Extract facts in the background.** A daemon thread is started running
   `extract_facts` against just this one exchange. It does not block your
   next prompt. `run_chat_loop` lightly reaps the previous turn's thread
   (`join(timeout=0)`) before starting a new one, and on program exit joins
   the final extraction thread with a 5-second timeout — long enough for
   most extraction calls to finish, but bounded so quitting never hangs.

The short-term sliding window (step 3) and long-term semantic recall
(step 2) are deliberately separate concerns: recent exchanges stay verbatim
in context; anything older survives only as a distilled, independently
retrievable fact.

## 7. Memory System Deep Dive

**Two memory tiers, one mechanism each:**

- *Short-term* — the actual message list, windowed by token budget. Exact,
  verbatim, but bounded and thread-local.
- *Long-term* — a flat table of independently-embedded fact strings, shared
  across every thread, retrieved by semantic similarity rather than recency
  or exact keyword match.

**Why brute-force cosine similarity instead of a vector database:** at the
scale of one person's remembered facts (realistically dozens to low
thousands of rows), a `numpy` linear scan over all embeddings takes single-digit
milliseconds. Adding a dedicated vector index would be premature
infrastructure for a single-user local tool — exactly the kind of complexity
this project was built to avoid.

**Dedup has two layers:** a new fact is rejected if it's too similar
(cosine similarity > 0.92) to *anything already in the database*, and also
if it's too similar to *another fact already inserted earlier in the same
extraction call* (the model is allowed to return several facts at once, and
near-duplicates among them are caught too — the in-memory `existing_embeddings`
list is updated after every insert specifically to catch this within-call
case).

**The extraction model call is fragile by nature, hardened defensively.**
Live testing showed the model intermittently wraps its JSON answer in
markdown code fences or prose. `_extract_json_array` strips fences and falls
back to slicing the first/last brackets before parsing, which resolves the
common cases; any output that still isn't parseable JSON, or any other
failure in the whole pipeline (model unreachable, malformed response, a
database error), is caught by `extract_facts`'s blanket exception handler
and logged to `extraction.log` rather than ever surfacing to the user or
crashing the background thread.

## 8. Configuration / Tunable Constants

All tunables are plain module-level constants at the top of `ochat.py` —
there is intentionally no separate config file. Edit the script directly to
change any of these:

| Constant | Value | Controls |
|---|---|---|
| `OLLAMA_URL` | `http://127.0.0.1:11434` | Ollama's local API base URL |
| `CHAT_MODEL` | `gemma4:12b` | The model used for chat and fact extraction |
| `EMBED_MODEL` | `nomic-embed-text` | The model used to embed facts and queries |
| `CONTEXT_TOKEN_BUDGET` | `8192` | Approximate token budget for the sliding-window history sent per turn |
| `CHARS_PER_TOKEN` | `4` | The token-estimation heuristic (`len(text) // 4`) |
| `RETRIEVAL_TOP_K` | `8` | Max number of long-term facts injected into the system prompt per turn |
| `RETRIEVAL_MIN_SIMILARITY` | `0.45` | Minimum cosine similarity for a fact to be considered relevant |
| `DEDUP_SIMILARITY_THRESHOLD` | `0.92` | Minimum cosine similarity for a new fact to be treated as a duplicate |
| `DEFAULT_THINK` | `"off"` | Default reasoning-effort level for new sessions |
| `EXTRACTION_JOIN_TIMEOUT_SECONDS` | `5` | How long `run_chat_loop` waits on exit for the last extraction thread to finish |
| `CALENDAR_ENABLED` | `True` | Master on/off switch for all calendar behavior (see §16) |
| `CALENDAR_LOOKAHEAD_DAYS` | `7` | How many days ahead `fetch_upcoming_events` looks |
| `CALENDAR_CACHE_TTL_SECONDS` | `300` | How long a fetched event list is reused before `refresh_calendar_cache` fetches again |
| `APPLESCRIPT_TIMEOUT_SECONDS` | `10` | Timeout (seconds) for any single `osascript` subprocess call |

## 9. CLI Reference

```
ochat                          # resume the "default" thread
ochat --thread <name>          # resume/create a named thread
ochat --think <level>          # off | on/true | low | medium | high | ... (passed through to Ollama)
ochat threads                  # list all threads with message counts and last-updated time
ochat memory list               # list every stored long-term fact
ochat memory forget <id>        # delete one fact by id
ochat calendar list             # list upcoming events from macOS Calendar.app (macOS only, see §16)
```

## 10. Error Handling & Resilience

| Failure | Behavior |
|---|---|
| Ollama not running | `check_ollama_ready` prints the exact fix (`ollama serve`) and exits before doing anything else |
| Required model not installed | Prints the exact `ollama pull <model>` command per missing model and exits — never pulls automatically |
| A required model is installed under Ollama's default `:latest` tag | Handled correctly by `_model_installed` — this exact bug was found and fixed during live testing |
| Thread file is corrupt/unparseable JSON | Renamed aside with a timestamp suffix, warning printed, fresh thread started; original bytes are preserved, never deleted |
| `save_thread` interrupted mid-write | Impossible to corrupt the target file — writes go to a temp file first, then an atomic `os.replace` |
| Chat or embedding request fails mid-turn | The turn aborts cleanly: nothing is appended to the thread, nothing is written to disk, you just retry |
| Long-term fact retrieval fails (e.g. a database error) | The turn does *not* abort — it proceeds with an empty fact list and a stderr warning |
| The extraction model returns non-JSON, fenced or prose-wrapped output | `_extract_json_array` recovers the JSON in the common cases; anything still unparseable is caught and logged, never crashes |
| Any other exception during background fact extraction | Caught by `extract_facts`'s blanket handler, logged to `extraction.log`, never propagates into the main interactive loop |

## 11. Concurrency Model

There is exactly one kind of concurrency in `ochat`: a single daemon
`threading.Thread` per turn, running `extract_facts`. The SQLite connection
(`init_db(..., check_same_thread=False)`) is shared between the main thread
and whichever extraction thread is currently running; WAL mode keeps reads
and writes from blocking each other. `run_chat_loop` only ever has at most
one extraction thread alive at a time — a new turn first does a
non-blocking `join(timeout=0)` to reap the previous one before starting the
next, and the program's `finally` block gives the very last one up to 5
seconds to finish before exiting. There is no thread pool, no shared
mutable state beyond the SQLite connection, and no locking required beyond
what SQLite's WAL mode already provides.

## 12. Testing

33 tests in `tests/test_memory.py`, run via:

```bash
uv run --with pytest --with numpy --with requests pytest tests/test_memory.py -v
```

The suite was built test-first (TDD) across the project's development:
pure-logic functions (similarity, dedup, truncation) are tested with plain
values and no mocking; storage functions use pytest's `tmp_path` fixture so
tests never touch the real `~/.local/share/ochat/` directory; all Ollama
HTTP calls are mocked with `unittest.mock`, asserting on actual request
payload shape (model name, JSON body, streaming flag) rather than just "a
call happened." Beyond unit tests, the full system was verified live against
a running Ollama instance with three end-to-end smoke tests: conversation
continuity across process restarts, fact extraction into `memory.db`, and
cross-thread semantic retrieval (a brand-new thread correctly recalling a
fact it was never told directly).

## 13. Known Limitations / Accepted Trade-offs

These were identified during development review and deliberately left as-is:

- `get_all_facts` returns read-only numpy arrays (`np.frombuffer`) — fine
  since nothing currently mutates a fact's embedding in place, but would
  need a `.copy()` if a future feature ever did.
- The CLI's `--help` output is minimal (no per-argument descriptions).
- `ochat threads` reuses `load_thread` purely to count messages, which means
  listing threads can silently quarantine a corrupt thread file as a side
  effect of what looks like a read-only command.
- Token counting is a `len(text) // 4` heuristic, not a real tokenizer —
  good enough for windowing decisions, not exact.
- No automated regression test exists for the narrow case where a required
  model name already has a tag but a *different* tag of the same base model
  is installed (e.g. `gemma4:12b` required, only `gemma4:27b` present) —
  hand-verified correct, but untested.

## 14. Development History

`ochat` was built from a written design spec and a 10-task TDD implementation
plan, executed task-by-task with an independent reviewer checking spec
compliance and code quality after every task, plus a final whole-branch
review at the end. One real bug was found during live end-to-end testing
(the Ollama `:latest` tagging issue) and fixed with its own regression test;
the final review surfaced two further hardening fixes (fenced/prose JSON
parsing, graceful retrieval-error containment) before the project was
considered complete. Source documents:

- `docs/superpowers/specs/2026-06-19-ochat-design.md` — the design spec
- `docs/superpowers/plans/2026-06-19-ochat-implementation.md` — the task-by-task implementation plan

## 15. Repository Layout

```
ochat/
  ochat.py                              # the entire application
  tests/test_memory.py                  # 33 tests
  .gitignore
  quickuse.md                           # quick reference (this doc's companion)
  Persememory.md                        # this file
  docs/superpowers/specs/...            # design spec
  docs/superpowers/plans/...            # implementation plan
```

## 16. Clock & Calendar Awareness

Added on top of the original design: `ochat` now tells the model the
current local date/time on every relevant call, and can optionally surface
and create events in macOS Calendar.app. The whole feature is additive — it
never slows down or changes behavior for a message that has nothing to do
with dates or calendars, and it degrades to a no-op anywhere Calendar.app
isn't available or accessible.

### Date/time awareness

| Function | Signature | Purpose |
|---|---|---|
| `current_datetime_context()` | `() -> str` | Returns `"Current date/time: <weekday>, <Month> <day>, <year>, <HH:MM AM/PM> <tz abbrev>"` using `datetime.now().astimezone()` — local time, not UTC. Pure function: no I/O beyond reading the system clock, no macOS dependency, so it keeps working on any platform and regardless of Calendar permission state |

Its output is injected in three places:

| # | Where | Effect |
|---|---|---|
| 1 | `build_system_prompt()` | Prepended to the base instruction on **every** chat turn, so the model always has a "now" anchor when answering |
| 2 | `EXTRACTION_PROMPT` (used by `extract_facts`) | Appended to the system prompt for the background fact-extraction call; the prompt text itself also instructs the model to resolve relative dates (e.g. "next Thursday") to absolute ones before recording a fact, so stored facts don't go stale once the original conversation scrolls out of context |
| 3 | `CALENDAR_INTENT_PROMPT` (used by `classify_calendar_intent`) | Passed as `now_context` so the calendar-intent classifier can resolve phrases like "tomorrow at 2pm" to an absolute ISO 8601 datetime |

Storage is unaffected: `created_at` and message `ts` fields remain UTC
everywhere, as before — only what's *told to the model* is local time.

### `ochat_calendar.py` — the macOS I/O layer

A new, separate file: a pure AppleScript (`osascript`) I/O layer with no
model calls, no prompts, and no CLI of its own — `ochat.py`'s orchestration
code decides when and why to call it, the same way it already calls
`insert_fact` or `ollama_embed`.

| Function / class | Signature | Purpose |
|---|---|---|
| `is_macos()` | `() -> bool` | `platform.system() == "Darwin"` — the single gate every calendar code path checks before doing anything |
| `CalendarError` | `Exception` subclass | Raised for any `osascript` failure: non-zero exit, timeout, or missing `osascript` binary |
| `fetch_upcoming_events(days_ahead, timeout)` | `(int, float) -> list[dict]` | Runs one AppleScript that filters every calendar's events by a `whose`-clause date range (`current date` to `current date + days_ahead`), avoiding an unfiltered scan (a known AppleScript Calendar-bridge performance trap). Returns `[{"title", "start" (ISO str), "end" (ISO str), "calendar"}, ...]`. Raises `CalendarError` on failure |
| `create_event(title, start, end, notes, timeout)` | `(str, datetime, datetime, str \| None, float) -> None` | Builds an AppleScript that starts from `current date` and overwrites its year/month/day/hour/minute fields individually (avoids AppleScript's locale-dependent date-string parsing), escapes `title`/`notes` for safe interpolation, and creates the event in `calendar 1`. Raises `CalendarError` on failure |
| `_run_applescript(script, timeout)` | private | Shells out via `subprocess.run(["osascript", "-e", script], ...)`; wraps `TimeoutExpired`/`FileNotFoundError` and any non-zero exit code into `CalendarError` |
| `_build_fetch_script(days_ahead)` / `_parse_events(raw)` | private | Construct the read AppleScript and parse its output. Fields are delimited with control characters unlikely to appear in real text — `chr(31)` (ASCII unit separator) between fields, `chr(30)` (record separator) between events — written into the script's own output construction rather than relying on AppleScript's native list/record serialization, which is far more fragile to parse from Python |
| `_escape_applescript_string(value)` / `_format_applescript_date_setup(var_name, when)` | private | Escape `"`/`\` before interpolating a string into generated AppleScript source; emit the `set year of ... / set month of ... / ...` statements used to build a date field-by-field |

### Read path: ambient calendar context with a TTL cache

`run_chat_loop` creates one cache dict per process —
`{"events": [...], "fetched_at": datetime | None}` — and threads it into
`handle_turn` as a 6th, optional `calendar_cache` parameter, the same way
`conn`/`thread`/`path` are already threaded through.

Each turn, if `calendar_cache` was supplied, `CALENDAR_ENABLED` is `True`,
and `ochat_calendar.is_macos()`, `handle_turn` calls
`refresh_calendar_cache(calendar_cache)`:

| Function | Purpose |
|---|---|
| `refresh_calendar_cache(calendar_cache)` | If `fetched_at` is set and younger than `CALENDAR_CACHE_TTL_SECONDS` (300s), returns the cached `events` list unchanged — no `osascript` call. Otherwise calls `ochat_calendar.fetch_upcoming_events(CALENDAR_LOOKAHEAD_DAYS, APPLESCRIPT_TIMEOUT_SECONDS)` and updates the cache. Any `CalendarError` is caught, a warning is printed to stderr, and the **last-known cached events** (or an empty list, on a first-ever failure) are returned — this fetch never aborts the turn, mirroring the existing fault-isolated fact-retrieval behavior in §6 |

When non-empty, the cached events are rendered as a new optional
`"Upcoming calendar events:"` bullet section in `build_system_prompt()`
(now `build_system_prompt(relevant_facts, calendar_events=None)`), alongside
the existing "Relevant memory" bullets — so the model can answer questions
like "what do I have on Thursday" using only ambient context, with no extra
model call.

### Write path: keyword gate → intent classification → confirmation

Creating an event is the one calendar behavior that costs extra latency, so
it's deliberately gated to only run for messages that plausibly need it:

| Step | Function | Purpose |
|---|---|---|
| 1. Cheap local gate | `looks_calendar_related(text)` | Lowercases the message and checks it against `CALENDAR_KEYWORDS` (`"calendar"`, `"schedule"`, `"scheduled"`, `"meeting"`, `"appointment"`, `"event"`, `"remind"`, `"reminder"`, the seven weekday names, `"tomorrow"`, `"tonight"`). Pure string logic, no I/O — this is what keeps ordinary chat turns exactly as fast as before the feature existed |
| 2. Intent classification | `classify_calendar_intent(user_input, now_context)` | Only called when step 1 matches. Sends `CALENDAR_INTENT_PROMPT` + `now_context` to `ollama_chat` (`think=False`, `stream_to_stdout=False`), parses the reply via `_extract_json_object()`, and returns `{"intent": "none"\|"query"\|"create", "title", "start", "end", "notes"}`. Any call/parse failure, or an `intent` value outside the three allowed strings, falls back to `{"intent": "none", ...}` — fails safe to "just a normal message" |
| 3. Orchestration | `handle_calendar_create_intent(user_input, now_context, calendar_cache)` | Calls step 2; if `intent != "create"` or `title`/`start`/`end` is missing, returns immediately (no-op). Otherwise parses `start`/`end` via `datetime.fromisoformat()` (returns on `ValueError`) and prints a confirmation prompt showing the **resolved absolute** date/time, e.g. `Add to calendar? "Dentist appointment" -- Thu, Jun 25 2026, 2:00-2:30 PM [y/N]`, then blocks on `input()`. Only `"y"`/`"yes"` (case-insensitive, after `.strip().lower()`) proceeds to call `ochat_calendar.create_event(...)`; anything else prints `"cancelled, nothing was added"` and returns. A `CalendarError` from `create_event` prints `error: failed to add event (...)` to stderr and returns — a failed or declined write never aborts the chat reply for that turn. On success, the new event is appended directly to `calendar_cache["events"]` so it's visible as context to the very same turn's chat reply (e.g. "add a meeting Thursday and tell me what else I have that day") |

This whole sequence runs inside `handle_turn`, **before** the main chat
call, in the same position long-term fact retrieval already occupies. An
`intent == "query"` result is intentionally a no-op beyond this point —
cached upcoming events (already ambient context every turn via the read
path above) are all that's used to answer calendar questions; there is no
separate query-triggered fetch.

### `ochat calendar list` — standalone CLI command

| Function | Purpose |
|---|---|
| `cmd_calendar_list()` | Bypasses the chat loop and the cache entirely. Exits with status 1 and a stderr message if not on macOS. Otherwise calls `ochat_calendar.fetch_upcoming_events(CALENDAR_LOOKAHEAD_DAYS, APPLESCRIPT_TIMEOUT_SECONDS)` directly and prints one line per event (`<start> - <end>  <title>  (<calendar>)`), or `"no upcoming events"`. Exits with status 1 and a stderr message on `CalendarError`. Doubles as a standalone way to confirm macOS Calendar/Automation permissions are actually granted, independent of chatting |

Wired into `main()` as a third subparser alongside `threads` and `memory`:
`ochat calendar list`.

### macOS-only graceful degradation

| Condition | Behavior |
|---|---|
| Not running on macOS (`ochat_calendar.is_macos()` is `False`) | All calendar behavior — read-path cache refresh, the keyword gate's downstream classification, `cmd_calendar_list` — silently no-ops or exits cleanly. `current_datetime_context()` keeps working everywhere, so date/time awareness in chat and fact extraction is unaffected |
| `osascript` missing, times out, or Calendar/Automation permission not yet granted | Both `fetch_upcoming_events` and `create_event` raise `CalendarError`. Reads degrade to the last-known cached events (or empty) plus a stderr warning; writes print a clear error and the turn continues normally. The one-time macOS permission dialog itself cannot be triggered or bypassed programmatically — the user has to approve it themselves (see `quickuse.md`) |
| Calendar-intent model call fails or returns unparseable JSON | `classify_calendar_intent` falls back to `{"intent": "none", ...}` — treated as an ordinary, non-calendar message |
| User declines the create-event confirmation | No write occurs; `"cancelled, nothing was added"` is printed; the turn continues exactly as if the message had been a normal chat message |

Design rationale and the original task-by-task implementation plan for this
feature are in `docs/superpowers/specs/2026-06-20-clock-calendar-design.md`
and `docs/superpowers/plans/2026-06-20-clock-calendar-implementation.md`.
