# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

`ochat.py` is a single-file Python CLI (~600 lines) that wraps a local Ollama
chat model with two things plain `ollama run` lacks: resumable named
conversation threads, and a long-term fact memory recalled by semantic search
across threads. It was built to replace a slower general-purpose agent
gateway, so every design choice favors speed and simplicity: no daemon, no
config file, no vector database — it only runs while actively in use.

It also gained clock/calendar awareness: every chat turn, fact-extraction
call, and calendar-intent classification is anchored to the current local
date/time, and on macOS it can surface upcoming Calendar.app events as
context and create new events from natural language (behind a y/N
confirmation). `ochat_calendar.py` is a second, separate file holding the
macOS-only AppleScript I/O for this. All of it degrades gracefully off macOS
or without Calendar permission — date/time awareness keeps working
regardless.

For full design rationale, data model, and the function-by-function
reference, read `Persememory.md` — it's exhaustive and should be treated as
the source of truth before re-deriving behavior from the code. `quickuse.md`
is the user-facing quick reference. `docs/superpowers/specs/` and
`docs/superpowers/plans/` hold the original design spec and TDD implementation
plan this project was built from.

## Commands

```bash
# Run the tool (uv resolves numpy + requests from the inline PEP 723 block
# automatically — no venv, no pip install)
./ochat.py
./ochat.py --thread work
./ochat.py --think low          # off | on/true | low | medium | high
./ochat.py threads
./ochat.py memory list
./ochat.py memory forget <id>
./ochat.py calendar list         # macOS only; lists upcoming Calendar.app events

# Run the full test suite (88 tests: tests/test_memory.py + tests/test_calendar.py)
uv run --with pytest --with numpy --with requests pytest tests/ -v

# Run a single test (from either tests/test_memory.py or tests/test_calendar.py)
uv run --with pytest --with numpy --with requests pytest tests/ -v -k test_name

# Prerequisite: Ollama must be running locally with both models pulled
ollama serve
ollama pull gemma4:12b
ollama pull nomic-embed-text
```

There is no separate lint/build step — it's a single dependency-declaring
script file.

## Architecture

Despite being one file, `ochat.py` is organized into four layers, top to
bottom in the source, and new code should stay within this layering:

1. **Pure logic** (no I/O) — `cosine_similarity`, `top_k_facts`,
   `is_duplicate_fact`, `estimate_tokens`, `truncate_messages_to_budget`,
   `current_datetime_context`, `looks_calendar_related`.
   Unit-test these with plain values, no mocking.
2. **Storage** — thread JSON files (`load_thread`/`save_thread`) and the
   SQLite fact store (`init_db`/`insert_fact`/`get_all_facts`/`delete_fact`).
   Filesystem/DB I/O, no network calls.
3. **Ollama client** — `check_ollama_ready`, `ollama_embed`, `ollama_chat`.
   All HTTP calls to the local Ollama server; mock `requests` in tests.
4. **Orchestration + CLI** — `handle_turn`, `run_chat_loop`, `extract_facts`,
   `refresh_calendar_cache`, `handle_calendar_create_intent`,
   `classify_calendar_intent`, and the `cmd_*`/`main()` argparse wiring.

There is also a **second I/O-layer file**, `ochat_calendar.py`, sitting
alongside `ochat.py`'s "Storage" and "Ollama client" layers but kept
separate because it's macOS-specific: a pure AppleScript (`osascript`) I/O
layer for Calendar.app, with `is_macos()`, `CalendarError`,
`fetch_upcoming_events(days_ahead, timeout)`, and
`create_event(title, start, end, notes, timeout)`. It has no model calls, no
prompts, and no CLI of its own — `ochat.py`'s orchestration layer decides
when and why to call it, the same way it already calls `insert_fact` or
`ollama_embed`. Always check `ochat_calendar.is_macos()` (or
`CALENDAR_ENABLED`) before calling into it, and expect/catch
`ochat_calendar.CalendarError` — it never raises anything else.

All runtime data lives outside the repo under `~/.local/share/ochat/`
(`threads/<name>.json`, `memory.db`, `extraction.log`) — never under the repo
itself.

### Two-tier memory model

- **Short-term**: the verbatim message list for the current thread, windowed
  per-turn to `CONTEXT_TOKEN_BUDGET` (8192 estimated tokens) by
  `truncate_messages_to_budget`.
- **Long-term**: a flat `facts` table in `memory.db`, shared across all
  threads, each row holding a fact string and its `nomic-embed-text`
  embedding (raw float32 bytes). Retrieved per-turn via brute-force cosine
  similarity (`top_k_facts`, top 8, min similarity 0.45) — no vector DB,
  since the working set is expected to stay in the hundreds-to-low-thousands
  range.

### Turn flow (`handle_turn`)

Embed input → retrieve top-k relevant facts (failure here is fault-isolated:
logs a warning and continues with an empty fact list, never aborts the turn)
→ build system prompt + windowed history → stream chat reply → atomically
persist thread (`save_thread` writes to `.tmp` then `os.replace`s) → spawn a
**daemon thread** running `extract_facts` against just that one exchange,
non-blocking. `run_chat_loop` reaps the previous turn's extraction thread
with `join(timeout=0)` before starting the next one, and on exit gives the
final extraction thread up to `EXTRACTION_JOIN_TIMEOUT_SECONDS` (5s) to
finish.

If the embed or chat HTTP call itself fails, the turn aborts completely —
nothing is appended to the thread and nothing is written to disk, so the user
can just resend the message.

### Fact extraction (`extract_facts`)

Runs in the background extraction thread; wrapped in a blanket
`except Exception` because it must **never** raise or crash that thread —
failures are written to `extraction.log` via `log_extraction_error` instead.
Model output is parsed as a JSON array of fact strings via
`_extract_json_array`, which strips markdown fences and falls back to
slicing between the first `[` and last `]` to tolerate prose-wrapped output.
New facts are deduped two ways before insertion: against every existing fact
embedding in `memory.db`, and against other facts already inserted earlier
in the *same* extraction call (the in-memory `existing_embeddings` list is
updated after each insert to catch within-call near-duplicates).

### Concurrency

Each turn can have up to two daemon threads in flight: `extract_facts` (at
most one alive at a time, reaped via the `pending_extraction` handle in
`run_chat_loop`) and, independently, a calendar-cache refresh thread spawned
by `refresh_calendar_cache` (at most one alive at a time, guarded by the
`calendar_cache["refreshing"]` flag — see "Calendar cache refresh is
fire-and-forget" below). Neither thread blocks the other, and neither blocks
the main chat turn. The SQLite connection is opened with
`check_same_thread=False` and WAL mode so the main thread and the extraction
thread can read/write without explicit locking.

### Calendar cache refresh is fire-and-forget

`refresh_calendar_cache` never blocks: it always returns
`calendar_cache`'s last-known `events` immediately. When the cache is stale
(or empty) and no refresh is already running, it flips
`calendar_cache["refreshing"]` to `True`, spawns a daemon thread
(`_run_calendar_refresh`, stashed at `calendar_cache["_thread"]`) to call
`ochat_calendar.fetch_upcoming_events`, and returns without waiting — the
*next* turn sees the refreshed cache, not the current one. This exists
because `fetch_upcoming_events`'s AppleScript `whose` filter on Calendar.app
events is a per-event round trip (tens of ms each) rather than a native
query, so against a real-world calendar set (many calendars, recurring
all-day calendars like Holidays/Birthdays) the call can legitimately take
60–90+ seconds — far longer than is acceptable to block an interactive chat
turn on, even though `APPLESCRIPT_TIMEOUT_SECONDS` (120s) is sized to let it
finish rather than error out. `ochat calendar list` (`cmd_calendar_list`)
intentionally bypasses this cache and calls `fetch_upcoming_events` directly
since it's a deliberate one-off command — it's expected to take as long as
the AppleScript call actually takes.

### Notable hardening details worth preserving when touching this code

- `_model_installed` handles Ollama's default `:latest` tagging: an untagged
  required model name (`nomic-embed-text`) matches any tagged variant
  installed; a required name with an explicit tag (`gemma4:12b`) must match
  exactly. This exact mismatch was a real bug found during live testing.
- `load_thread` never deletes a corrupt thread file — it renames it aside
  (`<name>.json.corrupt-<timestamp>`) and starts fresh.
- `check_ollama_ready` never auto-pulls a missing model; it always prints the
  exact `ollama pull <model>` command and exits.

### Configuration

All tunables are plain module-level constants at the top of `ochat.py` —
no config file. Alongside the original set (`OLLAMA_URL`, `CHAT_MODEL`,
`EMBED_MODEL`, `CONTEXT_TOKEN_BUDGET`, `CHARS_PER_TOKEN`, `RETRIEVAL_TOP_K`,
`RETRIEVAL_MIN_SIMILARITY`, `DEDUP_SIMILARITY_THRESHOLD`, `DEFAULT_THINK`,
`EXTRACTION_JOIN_TIMEOUT_SECONDS`), the calendar feature added:
`CALENDAR_ENABLED` (`True` — master on/off switch), `CALENDAR_LOOKAHEAD_DAYS`
(`7` — how far ahead `fetch_upcoming_events` looks), `CALENDAR_CACHE_TTL_SECONDS`
(`300` — how long `refresh_calendar_cache` reuses a fetched event list before
refreshing), and `APPLESCRIPT_TIMEOUT_SECONDS` (`120` — timeout for any single
`osascript` subprocess call in `ochat_calendar.py`; sized for `whose`-filtered
Calendar.app scans against real-world calendar volumes, not just `create_event`,
which is much faster). See `Persememory.md` §8 and §16 for the full reference.

## Testing conventions

Tests in `tests/test_memory.py` were built test-first (TDD): pure-logic
functions are tested with plain values, storage functions use pytest's
`tmp_path` fixture (never touch the real `~/.local/share/ochat/`), and all
Ollama HTTP calls are mocked with `unittest.mock.patch`, asserting on actual
request payload shape (model name, JSON body, streaming flag), not just that
a call happened. `tests/test_calendar.py` follows the same TDD/mocking
conventions (88 tests: tests/test_memory.py + tests/test_calendar.py): pure
logic (e.g. `looks_calendar_related`, `current_datetime_context`) is tested
directly, and `osascript` subprocess calls, `ollama_chat` intent
classification, and `builtins.input` confirmation prompts are all mocked so
no real Calendar.app, model, or terminal interaction is required.
