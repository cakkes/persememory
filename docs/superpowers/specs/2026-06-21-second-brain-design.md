# ochat — second brain indexing on the external drive

Date: 2026-06-21
Status: Approved for implementation

## Problem

The user's internal disk is nearly full (55GB free of 228GB) and most of
their documents, PDFs, and notes now live on a 2TB external APFS drive
(mounted at `/Volumes/JRAX82TB 1`), inside an existing Obsidian vault and
related folders under `EMMA BRAIN` (~410 files today: 288 PDFs, 74 docx, 28
doc, 7 pptx, 13 md, 6 mp3, 1 xlsx, plus a couple of Obsidian-native
`canvas`/`base` files and a few images). None of this content is currently
visible to `ochat` — it has no way to read, search, or add to it. The user
wants `ochat` to treat this folder as a "second brain": a free-text dump they
can write to (via Obsidian, Finder, or `ochat` itself) and retrieve from
conversationally, kept fresh automatically without manual re-indexing.

## Goals

- Read and write access to the external drive for `ochat`. (Verified during
  design: the drive mounts with `Owners: Disabled`, so any process running as
  the logged-in user — including `ochat` — already has full read/write
  access with no macOS TCC permission prompt, unlike Calendar.app. This is a
  path-configuration matter, not a permissions-engineering one.)
- Extract and semantically index the text content of every file currently
  under `EMMA BRAIN`, across all file types present today (pdf, doc, docx,
  pptx, xlsx, md, canvas, base, png/image via OCR, mp3 via transcription),
  excluding pure app/system plumbing (`.DS_Store`, `.obsidian/*`, dotfiles).
- Surface that indexed content **ambiently** during chat — ordinary
  questions get relevant second-brain context automatically, the same way
  fact memory and calendar context already work. No explicit search command.
- Let the user create new notes in the braindump via natural language in
  chat ("jot this down: ...") — written into the existing Obsidian
  inbox-capture convention (`EMMA BRAIN/Second Storage/0_Inbox`), gated by a
  y/N confirmation since it's a real write to the user's actual vault.
- Keep the index fresh automatically: detect new, edited, and deleted files
  and re-index every 8 hours via a scheduled background job — **without ever
  blocking an interactive chat turn** on a long-running scan. (This is a
  direct lesson from this same codebase's calendar-fetch fix: a `whose`-style
  full scan over hundreds of real files is a multi-second-to-multi-minute
  operation, and that cost must never land on a chat turn.)
- Preserve the project's existing philosophies: graceful degradation (a
  missing/empty index, or one bad file during reindex, never aborts
  anything), and no persistent daemon (the reindex is a one-shot process
  invoked on a schedule, not a long-running service).

## Non-goals

- No explicit search/browse command (e.g. `ochat brain search`) — retrieval
  is ambient-only, per explicit choice during design.
- No multi-folder configuration or drive auto-discovery by volume UUID — the
  target folder is a hardcoded path constant. If the drive ever remounts
  under a different name (it's already happened once: `JRAX82TB 1` is itself
  a disambiguation suffix), `ochat brain reindex` will cleanly error rather
  than silently operate on the wrong path, but won't auto-correct.
- No editing or deleting of *existing* files from chat — the write path only
  ever creates new files in the inbox. `ochat` never modifies or removes
  anything already in `EMMA BRAIN`.
- No cross-platform support — this feature requires macOS (Apple's PDFKit/
  Vision/Speech frameworks via PyObjC) and presumes the configured folder
  path is mounted. It does not degrade to an alternate mechanism off macOS;
  it simply doesn't run, the same way calendar features no-op off macOS.
- No real-time filesystem watching (FSEvents, etc.) — freshness comes from
  the 8-hour scheduled reindex plus an immediate single-file index right
  after an `ochat`-initiated write. A file dropped in via Finder/Obsidian
  mid-cycle isn't retrievable until the next scheduled or manual reindex.
- No parsing of Obsidian-specific semantics (backlinks, canvas node graphs,
  base queries) — `canvas`/`base` files are indexed as their raw JSON text,
  not structurally understood.
- No OCR/transcription quality post-processing — raw Vision/Speech framework
  output is stored as-is.

## Architecture

New file alongside `ochat.py` and `ochat_calendar.py`:

```
ochat.py                # adds brain orchestration, retrieval, write-intent flow
ochat_brain.py          # new -- folder scanning + per-file-type text extraction (macOS-native)
tests/test_memory.py    # existing
tests/test_calendar.py  # existing
tests/test_brain.py     # new
```

`ochat_brain.py` mirrors `ochat_calendar.py`'s role exactly: a pure I/O layer
with no model calls, no prompts, and no CLI of its own.

```python
class BrainExtractionError(Exception):
    """Raised by any extraction function on failure -- one error type per
    I/O layer, same pattern as CalendarError."""

def scan_folder(root: Path) -> list[dict]:
    """Walks root, skipping dotfiles/.obsidian/.git. Returns
    [{"path": Path, "mtime": float, "size": int}, ...] for every file."""

def extract_pdf_text(path: Path) -> str: ...
    # Apple PDFKit (Quartz). Raises BrainExtractionError.

def extract_office_text(path: Path) -> str: ...
    # Dispatches internally by suffix: .doc/.docx via the `textutil` CLI;
    # .pptx via python-pptx; .xlsx via openpyxl. Raises BrainExtractionError.

def extract_plain_text(path: Path) -> str: ...       # .md/.txt/.canvas/.base -- read as UTF-8 text
def extract_image_text(path: Path) -> str: ...       # Apple Vision OCR
def transcribe_audio(path: Path) -> str: ...          # Apple Speech framework

def write_note(folder: Path, title: str, content: str) -> Path:
    """Writes a new file named <YYYY-MM-DD-HHMM>.md into folder, with title
    rendered as an H1 heading followed by content. Returns the new path."""
```

New tunables in `ochat.py`, alongside the existing constants:

| Constant | Value | Controls |
|---|---|---|
| `BRAIN_ENABLED` | `True` | Master on/off switch, same role as `CALENDAR_ENABLED` |
| `BRAIN_FOLDER` | `Path("/Volumes/JRAX82TB 1/EMMA BRAIN")` | Root folder scanned for indexing |
| `BRAIN_INBOX_SUBFOLDER` | `"Second Storage/0_Inbox"` | Where `ochat`-written notes land, relative to `BRAIN_FOLDER` |
| `BRAIN_DB_PATH` | `DATA_DIR / "brain.db"` | SQLite index, separate from `memory.db` |
| `BRAIN_LOG_PATH` | `DATA_DIR / "brain.log"` | Per-file extraction failures during reindex (same role as `extraction.log`) |
| `BRAIN_CHUNK_SIZE_CHARS` | `1000` | Target chunk size for splitting extracted text |
| `BRAIN_CHUNK_OVERLAP_CHARS` | `150` | Overlap between consecutive chunks |
| `BRAIN_RETRIEVAL_TOP_K` | `5` | Max chunks injected into the system prompt per turn |
| `BRAIN_RETRIEVAL_MIN_SIMILARITY` | `0.45` | Same threshold as `RETRIEVAL_MIN_SIMILARITY` for facts |
| `BRAIN_KEYWORDS` | list | Keyword gate for the write-intent path (e.g. "remember this", "jot this down", "save this to my notes", "braindump") |

## Data model

A new `brain.db` (separate from `memory.db` — this is indexed *source
material*, not model-distilled facts), opened the same way (`check_same_thread=False`,
WAL mode):

```sql
CREATE TABLE brain_files (
    id INTEGER PRIMARY KEY,
    path TEXT NOT NULL UNIQUE,
    mtime REAL NOT NULL,
    size INTEGER NOT NULL,
    last_indexed_at TEXT NOT NULL
);

CREATE TABLE brain_chunks (
    id INTEGER PRIMARY KEY,
    file_id INTEGER NOT NULL REFERENCES brain_files(id),
    chunk_index INTEGER NOT NULL,
    text TEXT NOT NULL,
    embedding BLOB NOT NULL
);
```

Change detection uses `mtime` + `size` (a cheap `stat()`, no file content
read) rather than a content hash — sufficient for a personal, single-machine
use case, and avoids reading every file's bytes on every 8-hour check just to
detect "nothing changed."

## Extraction & chunking pipeline

```python
def chunk_text(text: str, size: int = BRAIN_CHUNK_SIZE_CHARS,
                overlap: int = BRAIN_CHUNK_OVERLAP_CHARS) -> list[str]: ...
```

Pure function, splits extracted text into overlapping chunks so a passage
spanning a chunk boundary isn't lost to retrieval — same spirit as
`truncate_messages_to_budget`'s windowing, applied to document text instead
of chat history.

Extension-to-extractor dispatch (in `ochat.py`, calling into `ochat_brain`):

| Extension(s) | Extractor |
|---|---|
| `.pdf` | `extract_pdf_text` |
| `.doc`, `.docx` | `extract_office_text` (via `textutil`) |
| `.pptx`, `.xlsx` | `extract_office_text` (via `python-pptx`/`openpyxl`) |
| `.md`, `.txt`, `.canvas`, `.base` | `extract_plain_text` |
| `.png`, `.jpg`, `.jpeg` | `extract_image_text` (OCR) |
| `.mp3`, `.m4a`, `.wav` | `transcribe_audio` |
| anything else | skipped, logged at debug level to `brain.log` |

## Reindex lifecycle

```python
def reindex_brain(conn, root: Path = BRAIN_FOLDER) -> dict:
    """Returns {"indexed": int, "removed": int, "skipped": int, "total_chunks": int}."""
```

1. `ochat_brain.scan_folder(root)` → current files on disk.
2. Diff against `brain_files`: new paths; paths whose `mtime`/`size` changed
   since last indexed; DB paths no longer present on disk.
3. For each new/changed file: delete its existing `brain_chunks` rows (if
   any), dispatch to the right extractor, `chunk_text` the result, embed each
   chunk with `ollama_embed`, insert chunk rows, upsert the `brain_files` row.
4. For each removed file: delete its chunk rows and its `brain_files` row.
5. **Every file is wrapped in its own try/except.** A single corrupt PDF, an
   OCR failure, or a transcription error logs a warning to `brain.log` and is
   skipped — it never aborts the run. Across ~410 real files, hitting a few
   bad ones (encrypted PDFs, etc.) is expected, not exceptional.

New CLI surface — exactly one command, reused for both the manual first-time
index and every scheduled run:

```
ochat brain reindex
```

Prints a one-line summary (e.g. `indexed 12, removed 3, skipped 1, 4821
chunks total`) and exits. If `BRAIN_FOLDER` doesn't exist (drive unmounted,
or remounted under a different name), it exits with a clear error message
rather than a raw traceback.

### Scheduling

A macOS `launchd` LaunchAgent (not cron — survives sleep/wake better, more
idiomatic on macOS), consistent with the project's "no persistent daemon"
rule: each firing is a short-lived process that runs `ochat brain reindex`
and exits.

`~/Library/LaunchAgents/com.ochat.brain-reindex.plist`:

```xml
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>com.ochat.brain-reindex</string>
    <key>ProgramArguments</key>
    <array>
        <string>/opt/homebrew/bin/uv</string>
        <string>run</string>
        <string>--script</string>
        <string>/Users/developer/ochat/ochat.py</string>
        <string>brain</string>
        <string>reindex</string>
    </array>
    <key>StartInterval</key>
    <integer>28800</integer>
    <key>StandardOutPath</key>
    <string>/Users/developer/.local/share/ochat/brain-reindex.stdout.log</string>
    <key>StandardErrorPath</key>
    <string>/Users/developer/.local/share/ochat/brain-reindex.stderr.log</string>
</dict>
</plist>
```

Absolute paths are required because `launchd` does not read the user's shell
profile (no `PATH` customization), so PATH-relative commands (plain `uv`,
plain `ochat`) would fail to resolve. Setup is: run `ochat brain reindex`
once manually to seed the initial index (this can legitimately take minutes
across ~410 files including OCR/transcription — same "deliberate one-off,
let it take as long as it takes" reasoning already applied to
`ochat calendar list`), then `launchctl load` the plist.

## Retrieval (ambient)

`handle_turn` already computes `query_embedding = ollama_embed(user_input)`
once per turn for fact retrieval. Brain-chunk retrieval reuses that same
embedding — no extra model call.

```python
def top_k_chunks(query_embedding, chunks, k=BRAIN_RETRIEVAL_TOP_K,
                  min_similarity=BRAIN_RETRIEVAL_MIN_SIMILARITY) -> list[dict]: ...
```

Pure function, same shape as `top_k_facts` (brute-force cosine similarity —
a few thousand chunk rows is still a single-digit-to-low-double-digit
millisecond numpy scan, well within the project's existing "no vector DB
needed yet" reasoning). `build_system_prompt` gains a third optional
section, "From your second brain:", alongside "Relevant memory" and
"Upcoming calendar events," listing chunk text with its source filename so
the model can cite where it came from.

If `brain.db` doesn't exist yet (feature never reindexed) or any read fails,
log a warning and continue with an empty chunk list — same fault-isolation
as fact retrieval; never aborts the turn.

## Write path (save a new note from chat)

Mirrors the calendar create-intent flow piece for piece:

```python
def looks_brain_related(text: str) -> bool: ...
```

Cheap, local, no-I/O keyword gate (`BRAIN_KEYWORDS`) — only on a match does
`handle_turn` pay for an extra model call.

```python
def classify_brain_intent(user_input: str, now_context: str) -> dict:
    """Returns {"intent": "save" | "none", "title": str | None, "content": str | None}.
    Any call/parse failure returns {"intent": "none", ...} -- fails safe."""
```

On `intent == "save"`, `handle_turn` calls:

```python
def handle_brain_save_intent(user_input: str, now_context: str, brain_conn) -> None: ...
```

If `title` or `content` is missing from the classifier's response, this is
treated as "not enough information to act on" and returns immediately with
no prompt — same fail-safe behavior as `handle_calendar_create_intent`'s
`if not title or not start_raw or not end_raw: return`. Otherwise it shows a
confirmation prompt previewing the note's title/content and target path
(`BRAIN_FOLDER / BRAIN_INBOX_SUBFOLDER`), blocks on `input()`, and only on
`y`/`yes` calls `ochat_brain.write_note(...)`. A decline, or any
`BrainExtractionError`/IO failure from the write itself, prints a clear
message and the turn continues normally. On success, that one new file is
immediately chunked, embedded, and inserted into `brain.db` directly (not a
full rescan) so it's retrievable later in the same conversation — mirroring
how a created calendar event immediately appears in that turn's cache.

This runs in the same position as the existing calendar create-intent check
inside `handle_turn` (after calendar, before the main chat call) — the two
keyword gates are independent and both cheap, so both simply run.

## Error handling & resilience (additions to the existing table)

| Failure | Behavior |
|---|---|
| `BRAIN_FOLDER` doesn't exist (drive unmounted/renamed) | `ochat brain reindex` exits with a clear error message; ambient retrieval during chat just sees an empty/stale `brain.db` and degrades gracefully (no crash) |
| A single file fails extraction (corrupt/encrypted/unsupported) | Logged to `brain.log`, file is skipped, reindex continues with the rest |
| `brain.db` missing, empty, or a read fails during a chat turn | Logged warning, empty chunk list, turn proceeds — same as fact-retrieval fault isolation |
| Brain-intent model call fails or returns unparseable JSON | Treated as `{"intent": "none"}` — fails safe to "just a normal chat message" |
| User declines the save-note confirmation | No write; a "cancelled, nothing was saved" message; turn continues |
| Write to the inbox folder fails (disk full, permissions changed, drive ejected mid-write) | Printed error, turn continues; nothing partially written (write then chunk/embed, in that order, so a failed write never reaches the indexing step) |

## Testing strategy

New `tests/test_brain.py`, following the existing project conventions:

- Pure-logic, no mocking: `chunk_text`, `looks_brain_related`, `top_k_chunks`
  — plain-value tests, same shape as the existing `top_k_facts` tests.
- Storage/change-detection tests with `tmp_path`: `brain_files`/`brain_chunks`
  schema round-trips; diff logic given fake `(path, mtime, size)` tuples
  against existing DB rows, covering new/changed/deleted classification.
- Extraction functions: mock the underlying PyObjC/`textutil` subprocess
  calls, same pattern as `_run_applescript`'s tests — asserting on what was
  invoked, not real OCR/transcription output.
- `reindex_brain`: mock `ochat_brain.scan_folder` and the extraction
  dispatch, verify new/changed/deleted handling end-to-end against a real
  tmp-path SQLite db, and verify a single extractor failure doesn't abort
  the rest of the run.
- Write-path flow: mirrors `test_handle_calendar_create_intent_*` exactly —
  mock `ollama_chat` and `builtins.input`, assert `ochat_brain.write_note`
  is/isn't called accordingly, and that a successful write is reflected in
  `brain.db` immediately.
- `cmd_brain_reindex`: mirrors the `test_cmd_calendar_list_*` patterns.
- The `launchd` plist itself is not unit-testable — verified operationally
  (`launchctl list`, a manual `launchctl start` / `launchctl kickstart`),
  same caveat already accepted for Calendar's permission-prompt dependency.

## Follow-ups (out of scope for this version)

- If OCR/transcription via PyObjC's Vision/Speech frameworks proves
  impractical once actually implemented (API friction, accuracy, or
  performance), fall back to indexing just the filename/path as a minimal
  placeholder for those files rather than blocking the rest of the feature.
- An explicit `ochat brain search <query>` command was considered and
  deferred (ambient-only was the explicit choice) — could be added later
  without touching the indexing/storage layer at all.
- Volume-UUID-based drive auto-discovery (so a remount under a different
  name doesn't require a manual constant change) was considered and deferred
  as unnecessary complexity for a single-user, single-machine tool.
