# ochat — clock & calendar awareness

Date: 2026-06-20
Status: Approved for implementation

## Problem

`ochat` records a UTC timestamp on every message and fact (`ts`/`created_at`),
but never tells the *model itself* what the current date/time is. The chat
model and the fact-extraction model both reason with no anchor to "now," so
they can't resolve relative references ("next Thursday", "in three weeks")
into absolute dates — which matters most for long-term facts, since a fact
stored as "has an appointment next Thursday" loses its meaning the moment
the conversation that produced it scrolls out of context. Separately, the
user has no way to surface their real calendar (macOS Calendar.app) inside a
chat session, or to add events to it without leaving the terminal.

## Goals

- Give the chat model, the fact-extraction model, and the new calendar-intent
  classifier a consistent "current local date/time" anchor, so relative date
  references can be resolved to absolute ones before being stored or acted on.
- Surface upcoming events from macOS Calendar.app (all calendars, next 7 days)
  as ambient context during chat.
- Allow creating calendar events from natural language in chat, gated by an
  explicit y/N confirmation showing the *resolved* absolute date/time, since
  this writes to a real, shared, hard-to-reverse system (the user's actual
  calendar).
- Preserve the project's core speed guarantee: ordinary chat turns that have
  nothing to do with the calendar must not pay any extra latency.
- Degrade gracefully off macOS, or wherever Calendar.app access isn't granted
  — date/time awareness must keep working even when calendar access doesn't.

## Non-goals

- No support for any calendar source other than macOS Calendar.app (no
  direct Google Calendar API, no ICS feeds) — Calendar.app already
  aggregates whatever accounts macOS itself is configured with.
- No per-calendar filtering — all calendars visible in Calendar.app are read.
- No event editing or deletion from chat — only reading upcoming events and
  creating new ones.
- No native Ollama tool/function-calling — intent detection reuses the
  existing "ask the model for JSON, parse defensively" pattern already used
  by fact extraction, for consistency and because tool-call support for
  `gemma4:12b` in Ollama is unverified.
- No EventKit/PyObjC integration in this version (see Follow-ups).

## Architecture

New file alongside `ochat.py`:

```
ochat.py                # adds clock helpers + calendar orchestration
ochat_calendar.py       # new -- macOS Calendar.app I/O layer (AppleScript)
tests/test_memory.py    # existing
tests/test_calendar.py  # new
```

`ochat_calendar.py` is a pure I/O layer, mirroring the existing "Storage" and
"Ollama client" layers already in `ochat.py`: it knows how to talk to
Calendar.app and nothing else — no model calls, no prompts, no CLI, no
business logic about when to call it. `ochat.py`'s orchestration layer
(`handle_turn` and friends) decides when and why to call it, exactly the way
it already calls `insert_fact` or `ollama_embed`.

New tunables, alongside the existing module-level constants in `ochat.py`:

| Constant | Value | Controls |
|---|---|---|
| `CALENDAR_ENABLED` | `True` | Master on/off switch for all calendar behavior |
| `CALENDAR_LOOKAHEAD_DAYS` | `7` | How far ahead `fetch_upcoming_events` looks |
| `CALENDAR_CACHE_TTL_SECONDS` | `300` | How long a fetched event list is reused before refreshing |
| `APPLESCRIPT_TIMEOUT_SECONDS` | `10` | Timeout for any single `osascript` subprocess call |

## Date/time awareness

A new pure-logic function in `ochat.py`:

```python
def current_datetime_context() -> str:
    now = datetime.now().astimezone()
    # e.g. "Current date/time: Saturday, June 20, 2026, 6:51 PM PDT (America/Los_Angeles)"
```

This has no macOS dependency and no I/O beyond reading the system clock — it
keeps working regardless of platform or Calendar.app permission state. Its
output is prepended to three places:

1. `build_system_prompt` — every chat turn.
2. `EXTRACTION_PROMPT` — the fact-extraction call, whose prompt text is also
   updated to explicitly instruct: *"Resolve relative dates (e.g. 'next
   Thursday') to absolute dates before recording a fact."*
3. The new `CALENDAR_INTENT_PROMPT` (below).

Storage stays UTC everywhere (`created_at`, `ts` — unchanged); only what's
told to the model is local time. No schema or stored-format changes.

## Calendar read path

`ochat_calendar.py`:

```python
def is_macos() -> bool: ...

class CalendarError(Exception): ...

def fetch_upcoming_events(days_ahead: int, timeout: float) -> list[dict]:
    """Returns [{"title": str, "start": iso str, "end": iso str, "calendar": str}, ...]
    Raises CalendarError on any osascript failure, timeout, or unparseable output."""
```

Implementation runs a single AppleScript that filters Calendar.app's events
by a date range (`whose` clause) across every calendar, rather than
enumerating everything — AppleScript's Calendar bridge is well known to be
slow on large calendars, so an unfiltered scan is the main read-path
performance risk this design avoids. Output is parsed using a delimiter
unlikely to appear in real text — a control character such as the ASCII
unit-separator (decimal 31) between fields and the record-separator
(decimal 30) between events — written into the AppleScript's own output
construction, not AppleScript's native list/record serialization. This is
far less fragile to parse from Python than the default record format.

`ochat.py`'s `run_chat_loop` creates one cache dict per process,
`{"events": [...], "fetched_at": datetime | None}`, and threads it into
`handle_turn` the same way `conn`/`thread`/`path` already are. Each turn, if
`CALENDAR_ENABLED`, `ochat_calendar.is_macos()`, and the cache is missing or
older than `CALENDAR_CACHE_TTL_SECONDS`, `handle_turn` calls
`fetch_upcoming_events`. Any `CalendarError` (permission denied, `osascript`
missing, timeout) is caught, logged as a warning (same style as the existing
fact-retrieval fault-isolation), and the turn proceeds with whatever was
last cached (or empty, on the very first failure) — never aborts the turn.

When non-empty, cached events are rendered as a new bullet section in the
system prompt (`build_system_prompt` gains an optional `calendar_events`
parameter), alongside the existing "Relevant memory" bullets.

## Calendar write path

A new pure-logic gate in `ochat.py`, fully unit-testable with plain strings:

```python
def looks_calendar_related(text: str) -> bool: ...
```

A cheap, local, no-I/O keyword/regex scan (calendar/meeting/appointment/
schedule/remind/event + weekday names, "tomorrow", "next week", etc.). Only
when this returns `True` does `handle_turn` pay for an extra model call —
ordinary chat turns are completely unaffected, preserving the speed
guarantee.

On a match, `handle_turn` calls a new orchestration function:

```python
def classify_calendar_intent(user_input: str, now_context: str) -> dict:
    """Calls ollama_chat with CALENDAR_INTENT_PROMPT (think=False,
    stream_to_stdout=False); returns
    {"intent": "none" | "query" | "create", "title": str | None,
     "start": str | None, "end": str | None, "notes": str | None}.
    Any failure to call or parse returns {"intent": "none", ...}."""
```

`CALENDAR_INTENT_PROMPT` is a fixed system prompt (same shape as
`EXTRACTION_PROMPT`) asking for exactly one JSON object, given
`now_context`, so relative dates resolve correctly. Parsing reuses the
existing fence-stripping/bracket-slicing trick from `_extract_json_array`,
generalized into a shared helper that can slice between either `[`/`]` or
`{`/`}` so both call sites share one implementation.

- **`intent == "create"`**: `handle_turn` prints a confirmation showing the
  **resolved absolute** start/end (not the user's original phrase), e.g.:

  ```
  Add to calendar? "Dentist appointment" -- Thu, Jun 25 2026, 2:00-2:30 PM [y/N]
  ```

  and blocks on `input()`. Only on `y`/`yes` (case-insensitive) does it call
  `ochat_calendar.create_event(...)`. A decline, or any `CalendarError` raised
  by `create_event`, prints a clear message and the turn continues normally
  — a failed or declined write never aborts the chat reply for that turn.
- **`intent == "query"`**: falls through to using whatever's in the read-path
  cache (§ above) for context; no separate fetch.
- **`intent == "none"`**, or classification itself fails/errors: behaves
  exactly as today.

This step runs **before** the main chat call, in the same position fact
retrieval already occupies, so a created event is visible to the model
answering that same turn (e.g. "add a meeting Thursday and tell me what else
I have that day").

`ochat_calendar.py` adds:

```python
def create_event(title: str, start: datetime, end: datetime,
                  notes: str | None, timeout: float) -> None:
    """Raises CalendarError on any osascript failure."""
```

Event dates are constructed in the AppleScript by taking `current date` and
overwriting its year/month/day/hour/minute fields individually, not by
parsing an AppleScript date-string literal — date-string parsing in
AppleScript is locale-dependent and a well-known source of subtle bugs.
String fields (`title`, `notes`) are escaped (quotes/backslashes) before
being interpolated into the generated AppleScript source, the same caution
warranted any time a value is interpolated into a command string passed to
a subprocess.

## CLI

`ochat calendar list` — read-only dump of upcoming events (bypasses the chat
loop and the cache entirely; calls `fetch_upcoming_events` directly), mirroring
`ochat memory list` / `ochat threads`. Doubles as a way to confirm macOS
Calendar permissions are actually granted, independent of chatting.

## Error handling & resilience (additions to the existing table)

| Failure | Behavior |
|---|---|
| Not running on macOS, or `osascript` unavailable | `is_macos()` is `False`; all calendar behavior silently no-ops. Date/time awareness still works. |
| Calendar/Automation permission not yet granted or denied | `fetch_upcoming_events`/`create_event` raise `CalendarError`; reads degrade to "no events" with a logged warning, writes print a clear error and are skipped. The one-time macOS permission dialog itself can't be triggered or bypassed programmatically — `quickuse.md` will document approving it once. |
| `osascript` call times out (e.g. very large calendar) | Treated as a `CalendarError` — same degrade-gracefully handling as above. |
| Calendar-intent model call fails or returns unparseable JSON | Treated as `{"intent": "none"}` — fails safe to "just a normal chat message." |
| User declines the create-event confirmation | No write; "cancelled, nothing was added" message; turn continues. |

## Testing strategy

New `tests/test_calendar.py`, following the existing project conventions
exactly:

- `looks_calendar_related` and `current_datetime_context`: pure-logic tests
  with plain strings, no mocking.
- `fetch_upcoming_events` / `create_event`: mock `ochat_calendar.subprocess.run`
  (or equivalent), asserting on the constructed AppleScript source's shape
  (e.g. that the date-range bounds and escaped title appear in it) — never a
  real Calendar.app call, so tests run on any OS/CI.
- `classify_calendar_intent`: mock `ochat.ollama_chat`'s return value, same
  pattern as the existing `extract_facts` tests (including fenced/prose JSON
  and malformed-JSON cases).
- Confirmation flow: mock `builtins.input` to return `"y"` / `"n"` and assert
  `ochat_calendar.create_event` is/isn't called accordingly.

## Follow-ups (out of scope for this version)

- If AppleScript read latency proves too slow in practice even with
  date-range filtering, replace `ochat_calendar.py`'s internals with
  PyObjC + EventKit (`pyobjc-framework-EventKit`) behind the same function
  signatures — `fetch_upcoming_events`/`create_event`'s call sites in
  `ochat.py` would not need to change.
- Editing/deleting events, or supporting calendar sources other than
  Calendar.app, are explicitly deferred.
