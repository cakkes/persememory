# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

`ochat.py` is a single-file Python CLI (~510 lines) that wraps a local Ollama
chat model with two things plain `ollama run` lacks: resumable named
conversation threads, and a long-term fact memory recalled by semantic search
across threads. It was built to replace a slower general-purpose agent
gateway, so every design choice favors speed and simplicity: no daemon, no
config file, no vector database — it only runs while actively in use.

It also gained clock awareness: every chat turn and fact-extraction call is
anchored to the current local date/time, so relative references ("next
Thursday") can be resolved before being stored or acted on.

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

# Run the full test suite (44 tests: tests/test_memory.py)
uv run --with pytest --with numpy --with requests pytest tests/ -v

# Run a single test
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
   `effective_history_budget`, `current_datetime_context`.
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

### Deployed clone — this repo is NOT what `ochat` runs

This working directory is the dev repo. The `ochat` command on `$PATH` is
`~/.local/bin/ochat`, a symlink into a **separate clone** at
`/Users/developer/ochat/` (same GitHub remote, independent checkout). A fix
committed and merged here does nothing for the user's actual running tool
until it's pushed to origin and pulled into that other clone — this already
caused one real incident (2026-06-21: a context-truncation bug was fixed
and merged here, but the user kept hitting it because the deployed clone
was still 5 commits behind).

`scripts/git-hooks/post-commit` and `post-merge` (wired up via
`git config core.hooksPath scripts/git-hooks`, already set in this clone)
auto-push `main` to origin and `git pull --ff-only` it into
`/Users/developer/ochat/` whenever main's tip moves here, specifically so
this can't recur. A *new* clone of this repo won't have `core.hooksPath`
set automatically (git never auto-runs hooks from a fresh clone, by
design) — run that `git config` command once after cloning to enable it.

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
→ build system prompt + windowed history (sized by `effective_history_budget`)
→ stream chat reply, retrying once with a smaller window if it gets cut off
→ atomically persist thread (`save_thread` writes to `.tmp` then `os.replace`s) → spawn a
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

Each turn has at most one daemon thread in flight: `extract_facts` (reaped
via the `pending_extraction` handle in `run_chat_loop`), which never blocks
the main chat turn. The SQLite connection is opened with
`check_same_thread=False` and WAL mode so the main thread and the extraction
thread can read/write without explicit locking.

### Notable hardening details worth preserving when touching this code

- `_model_installed` handles Ollama's default `:latest` tagging: an untagged
  required model name (`nomic-embed-text`) matches any tagged variant
  installed; a required name with an explicit tag (`gemma4:12b`) must match
  exactly. This exact mismatch was a real bug found during live testing.
- `load_thread` never deletes a corrupt thread file — it renames it aside
  (`<name>.json.corrupt-<timestamp>`) and starts fresh.
- `check_ollama_ready` never auto-pulls a missing model; it always prints the
  exact `ollama pull <model>` command and exits.
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

### Configuration

All tunables are plain module-level constants at the top of `ochat.py` —
no config file: `OLLAMA_URL`, `CHAT_MODEL`, `EMBED_MODEL`,
`CONTEXT_TOKEN_BUDGET`, `OLLAMA_NUM_CTX`, `RESPONSE_TOKEN_RESERVE`,
`CHARS_PER_TOKEN`, `RETRIEVAL_TOP_K`, `RETRIEVAL_MIN_SIMILARITY`,
`DEDUP_SIMILARITY_THRESHOLD`, `DEFAULT_THINK`,
`EXTRACTION_JOIN_TIMEOUT_SECONDS`. See `Persememory.md` §8 for the full
reference.

## Testing conventions

Tests in `tests/test_memory.py` (44 tests) were built test-first (TDD):
pure-logic functions are tested with plain values, storage functions use
pytest's `tmp_path` fixture (never touch the real `~/.local/share/ochat/`),
and all Ollama HTTP calls are mocked with `unittest.mock.patch`, asserting
on actual request payload shape (model name, JSON body, streaming flag),
not just that a call happened.
