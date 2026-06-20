# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

`ochat.py` is a single-file Python CLI (~440 lines) that wraps a local Ollama
chat model with two things plain `ollama run` lacks: resumable named
conversation threads, and a long-term fact memory recalled by semantic search
across threads. It was built to replace a slower general-purpose agent
gateway, so every design choice favors speed and simplicity: no daemon, no
config file, no vector database — it only runs while actively in use.

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

# Run the full test suite (33 tests)
uv run --with pytest --with numpy --with requests pytest tests/test_memory.py -v

# Run a single test
uv run --with pytest --with numpy --with requests pytest tests/test_memory.py -v -k test_name

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
   `is_duplicate_fact`, `estimate_tokens`, `truncate_messages_to_budget`.
   Unit-test these with plain values, no mocking.
2. **Storage** — thread JSON files (`load_thread`/`save_thread`) and the
   SQLite fact store (`init_db`/`insert_fact`/`get_all_facts`/`delete_fact`).
   Filesystem/DB I/O, no network calls.
3. **Ollama client** — `check_ollama_ready`, `ollama_embed`, `ollama_chat`.
   All HTTP calls to the local Ollama server; mock `requests` in tests.
4. **Orchestration + CLI** — `handle_turn`, `run_chat_loop`, `extract_facts`,
   and the `cmd_*`/`main()` argparse wiring.

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

Exactly one daemon thread per turn (`extract_facts`), at most one alive at a
time. The SQLite connection is opened with `check_same_thread=False` and WAL
mode so the main thread and the extraction thread can read/write without
explicit locking.

### Notable hardening details worth preserving when touching this code

- `_model_installed` handles Ollama's default `:latest` tagging: an untagged
  required model name (`nomic-embed-text`) matches any tagged variant
  installed; a required name with an explicit tag (`gemma4:12b`) must match
  exactly. This exact mismatch was a real bug found during live testing.
- `load_thread` never deletes a corrupt thread file — it renames it aside
  (`<name>.json.corrupt-<timestamp>`) and starts fresh.
- `check_ollama_ready` never auto-pulls a missing model; it always prints the
  exact `ollama pull <model>` command and exits.

## Testing conventions

Tests in `tests/test_memory.py` were built test-first (TDD): pure-logic
functions are tested with plain values, storage functions use pytest's
`tmp_path` fixture (never touch the real `~/.local/share/ochat/`), and all
Ollama HTTP calls are mocked with `unittest.mock.patch`, asserting on actual
request payload shape (model name, JSON body, streaming flag), not just that
a call happened.
