# ochat — persistent-memory terminal chat for Ollama

Date: 2026-06-19
Status: Approved for implementation

## Problem

The user runs `gemma4:12b` locally via Ollama and chats with it directly through
`ollama run gemma4:12b`. That gives zero persistence: closing the terminal loses
the conversation, and there's no memory of facts/preferences across sessions.
OpenClaw previously filled this role but was uninstalled for being too slow.

## Goals

- Resume a named conversation thread across terminal sessions (conversation
  continuity).
- Recall specific facts/preferences from arbitrarily old conversations, even
  once they've scrolled out of the active context window (long-term memory),
  via semantic search rather than keyword matching, since the fact store is
  expected to grow large over time.
- Stay fast. The OpenClaw complaint was latency (hidden "thinking" tokens,
  cold-start reloads) — this tool must not reintroduce that. No background
  service/daemon; it only runs while invoked. Default thinking mode is `off`.
- Stay simple and inspectable: plain files (JSON, SQLite) the user can read or
  edit directly, no config-file layer, no extra running services.

## Non-goals

- No multi-device sync.
- No GUI.
- No automatic pulling of models without explicit user action.
- No support for remote/non-Ollama model providers.

## Architecture

Self-contained git repo at `~/ochat/`:

```
~/ochat/
  ochat.py          # the whole tool (chat loop + memory)
  tests/
    test_memory.py  # unit tests for similarity/dedup/truncation logic
  docs/superpowers/specs/2026-06-19-ochat-design.md
```

`ochat.py` uses a `uv run --script` shebang with inline PEP 723 dependency
metadata (`numpy`, `requests`), so `uv` transparently manages an ephemeral
dependency environment — no manual venv/requirements step. The file is
`chmod +x` and symlinked to `~/.local/bin/ochat` (already on `PATH`), so the
user types `ochat` instead of `ollama run gemma4:12b`.

Data lives outside the repo, under `~/.local/share/ochat/`:

- `threads/<name>.json` — one file per named conversation thread, full
  history, never auto-pruned.
- `memory.db` — single SQLite database (WAL mode) holding the long-term fact
  store, shared across all threads.
- `extraction.log` — append-only log for background fact-extraction errors.

New Ollama dependency: `nomic-embed-text` (~280MB), pulled separately from
`gemma4:12b`, used only to embed facts and queries for semantic search. Chosen
for its small footprint given the host already shows some swap pressure with
`gemma4:12b` loaded.

No config file. Tunable values (model names, context window size, retrieval
count, similarity thresholds) are constants at the top of `ochat.py`, editable
directly.

## Data model

SQLite (`memory.db`):

```sql
CREATE TABLE facts (
  id INTEGER PRIMARY KEY,
  text TEXT NOT NULL,
  embedding BLOB NOT NULL,     -- float32 vector from nomic-embed-text
  source_thread TEXT,
  created_at TEXT NOT NULL
);
```

Thread file (`threads/<name>.json`):

```json
{
  "name": "work",
  "messages": [
    {"role": "user", "content": "...", "ts": "..."},
    {"role": "assistant", "content": "...", "ts": "..."}
  ]
}
```

## Turn-by-turn flow

1. User runs `ochat` (default thread) or `ochat --thread <name>`.
2. Load (or create) the named thread's JSON.
3. On each user message:
   a. Embed the message via `nomic-embed-text`.
   b. Brute-force cosine-similarity scan over all rows in `memory.db`
      (numpy; no vector DB needed at personal scale): compute similarity for
      every fact, discard any below a 0.45 cutoff, then take the top 8 of
      what remains (fewer than 8 if fewer qualify).
   c. Build the system prompt: base instructions + a "Relevant memory" bullet
      list from (b).
   d. Build the API payload: system prompt + a sliding window of the most
      recent thread messages trimmed to fit an 8192-token context budget +
      the new user message. Token count is approximated with a simple
      character-based heuristic (~4 chars/token) — no real tokenizer
      dependency. Full history always stays on disk; only the live payload
      is windowed.
   e. Call Ollama `/api/chat` with `think:false` by default (overridable via
      `--think <level>`), streaming the reply to the terminal.
   f. Persist the updated thread to disk immediately (write-to-temp-then-
      rename, so a kill -9 mid-turn can't corrupt it).
   g. In a background thread (non-blocking — doesn't delay the next prompt):
      call `gemma4:12b` with the last exchange, asking it to extract any new
      durable facts as a JSON array (or `[]`). Each extracted fact is
      embedded; if it's >0.92 cosine-similar to an existing fact it's skipped
      (dedup), otherwise inserted into `memory.db`.
4. On exit, the program joins any in-flight extraction thread with a short
   timeout (5s) so a quick quit right after the final reply doesn't usually
   lose that exchange's facts — but it does not block indefinitely, so an
   extraction call that's unusually slow can still be missed on exit. This is
   an accepted limitation, not a bug to fix later.

The short-term sliding window (3d) and long-term semantic recall (3b) are
deliberately separate: recent exchanges stay verbatim in context; older
exchanges survive only as distilled, retrievable facts.

## CLI

- `ochat` — resume the default thread interactively.
- `ochat --thread <name>` — resume/create a named thread.
- `ochat --think <level>` — override thinking level for this run
  (`off|low|medium|high`, etc., matching Ollama's accepted values; default
  `off`).
- `ochat threads` — list threads with message counts and last-updated time.
- `ochat memory list` — view stored long-term facts (id, text, created date,
  source thread). Included so the user can audit what an automatic-extraction
  system has decided to remember.
- `ochat memory forget <id>` — delete a specific fact by id.

## Error handling

- Startup pings Ollama (`/api/version`) and checks both `gemma4:12b` and
  `nomic-embed-text` are present via `/api/tags`. If the embedding model is
  missing, prints the exact `ollama pull nomic-embed-text` command and exits
  — never auto-pulls without the user running it themselves.
- A corrupt/unparseable thread JSON file is renamed aside
  (`<name>.json.corrupt-<timestamp>`) and a fresh thread starts, with a
  warning printed. Never silently overwritten or lost.
- A failed chat API call prints an error and writes nothing to thread
  history, so the user can simply retry.
- Background fact-extraction errors (model call failure, SQLite error) are
  caught, logged to `extraction.log`, and never surfaced as a crash or stall
  in the main interactive loop.
- SQLite errors during retrieval are caught; that turn proceeds with "no
  relevant facts" rather than crashing the chat.

## Testing strategy

Unit tests (pytest via `uv run pytest`), all runnable without a live Ollama
instance using mocked embeddings/model responses:

- Cosine-similarity ranking / top-k selection.
- Dedup threshold logic (fact rejected if too similar to an existing one).
- Sliding-window truncation given a long message list.
- Thread JSON load/save round-trip, including the corrupt-file recovery path.

Manual smoke test after implementation (not automated — not worth it for a
single-user interactive tool):

1. New thread → exchange a few messages → quit → resume same thread → confirm
   continuity.
2. Trigger fact extraction → confirm the fact appears via `ochat memory list`.
3. Ask an unrelated-seeming later question that should surface that fact via
   retrieval → confirm it's included in the model's context.
