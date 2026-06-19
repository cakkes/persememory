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

## 9. CLI Reference

```
ochat                          # resume the "default" thread
ochat --thread <name>          # resume/create a named thread
ochat --think <level>          # off | on/true | low | medium | high | ... (passed through to Ollama)
ochat threads                  # list all threads with message counts and last-updated time
ochat memory list               # list every stored long-term fact
ochat memory forget <id>        # delete one fact by id
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
