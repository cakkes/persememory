# Clock & Calendar Awareness Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Give `ochat`'s chat model, fact-extraction model, and a new calendar-intent classifier a current local date/time anchor, and add read/write access to macOS Calendar.app (upcoming events as ambient context, event creation gated by a heuristic keyword filter plus an explicit y/N confirmation).

**Architecture:** New `ochat_calendar.py` module is a pure AppleScript I/O layer (no model calls, no prompts) — `is_macos()`, `CalendarError`, `fetch_upcoming_events()`, `create_event()`. All orchestration (when to call it, how to classify intent, the confirmation flow) lives in `ochat.py`, following the project's existing four-layer split. `handle_turn` gains an optional `calendar_cache` parameter threaded through from `run_chat_loop`, mirroring how `conn`/`thread`/`path` are already passed.

**Tech Stack:** Python 3.11+ stdlib only for the new module (`subprocess`, `platform`, `datetime`) — no new pip/uv dependencies. AppleScript via `osascript`. Existing `pytest` + `unittest.mock` conventions.

## Global Constraints

- No new dependencies — `ochat.py`'s PEP 723 block stays `["numpy", "requests"]`; `ochat_calendar.py` uses only stdlib.
- `CALENDAR_ENABLED = True`, `CALENDAR_LOOKAHEAD_DAYS = 7`, `CALENDAR_CACHE_TTL_SECONDS = 300`, `APPLESCRIPT_TIMEOUT_SECONDS = 10` — exact values from the spec.
- No SQLite schema changes; resolved absolute dates land in the existing `facts.text` column as plain text.
- All calendar behavior must no-op gracefully off macOS or when Calendar permission is denied — date/time awareness must keep working regardless.
- A calendar write only happens after an explicit `y`/`yes` confirmation showing the **resolved absolute** date/time.
- Every new function that does I/O (subprocess, model calls) must be unit-testable via mocking, with no test requiring a live Ollama instance or real Calendar.app access.
- Spec source of truth: `docs/superpowers/specs/2026-06-20-clock-calendar-design.md`.

---

### Task 1: `current_datetime_context()` pure helper

**Files:**
- Modify: `ochat.py` (add function after `truncate_messages_to_budget`, ochat.py:76)
- Test: `tests/test_calendar.py` (new file)

**Interfaces:**
- Produces: `ochat.current_datetime_context() -> str`

- [ ] **Step 1: Write the failing test**

Create `tests/test_calendar.py`:

```python
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import ochat


def test_current_datetime_context_formats_local_time():
    fixed = datetime(2026, 6, 20, 18, 51, tzinfo=timezone(timedelta(hours=-7)))
    with patch("ochat.datetime") as mock_datetime:
        mock_datetime.now.return_value.astimezone.return_value = fixed
        result = ochat.current_datetime_context()
    assert result == "Current date/time: Saturday, June 20, 2026, 06:51 PM UTC-07:00"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run --with pytest --with numpy --with requests pytest tests/test_calendar.py -v`
Expected: FAIL with `AttributeError: module 'ochat' has no attribute 'current_datetime_context'`

- [ ] **Step 3: Write minimal implementation**

In `ochat.py`, immediately after `truncate_messages_to_budget` (after line 76, before `def thread_path`):

```python
def current_datetime_context() -> str:
    now = datetime.now().astimezone()
    return f"Current date/time: {now.strftime('%A, %B %d, %Y, %I:%M %p %Z')}"
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run --with pytest --with numpy --with requests pytest tests/test_calendar.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add ochat.py tests/test_calendar.py
git commit -m "feat: add current_datetime_context helper"
```

---

### Task 2: `looks_calendar_related()` pure keyword gate

**Files:**
- Modify: `ochat.py` (add constant + function after `current_datetime_context`)
- Test: `tests/test_calendar.py`

**Interfaces:**
- Produces: `ochat.looks_calendar_related(text: str) -> bool`, `ochat.CALENDAR_KEYWORDS`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_calendar.py`:

```python
def test_looks_calendar_related_true_for_meeting_keyword():
    assert ochat.looks_calendar_related("add a meeting tomorrow at 2pm") is True


def test_looks_calendar_related_true_for_weekday_name():
    assert ochat.looks_calendar_related("let's talk Thursday") is True


def test_looks_calendar_related_is_case_insensitive():
    assert ochat.looks_calendar_related("REMIND me about this") is True


def test_looks_calendar_related_false_for_unrelated_text():
    assert ochat.looks_calendar_related("what's the weather like") is False
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run --with pytest --with numpy --with requests pytest tests/test_calendar.py -v`
Expected: FAIL with `AttributeError: module 'ochat' has no attribute 'looks_calendar_related'`

- [ ] **Step 3: Write minimal implementation**

In `ochat.py`, after `current_datetime_context`:

```python
CALENDAR_KEYWORDS = (
    "calendar", "schedule", "scheduled", "meeting", "appointment", "event",
    "remind", "reminder", "monday", "tuesday", "wednesday", "thursday",
    "friday", "saturday", "sunday", "tomorrow", "tonight",
)


def looks_calendar_related(text: str) -> bool:
    lowered = text.lower()
    return any(keyword in lowered for keyword in CALENDAR_KEYWORDS)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run --with pytest --with numpy --with requests pytest tests/test_calendar.py -v`
Expected: PASS (5 tests total)

- [ ] **Step 5: Commit**

```bash
git add ochat.py tests/test_calendar.py
git commit -m "feat: add looks_calendar_related keyword gate"
```

---

### Task 3: `ochat_calendar.py` skeleton — `CalendarError`, `is_macos()`

**Files:**
- Create: `ochat_calendar.py`
- Test: `tests/test_calendar.py`

**Interfaces:**
- Produces: `ochat_calendar.CalendarError(Exception)`, `ochat_calendar.is_macos() -> bool`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_calendar.py`:

```python
import ochat_calendar


def test_calendar_error_is_an_exception():
    assert issubclass(ochat_calendar.CalendarError, Exception)


def test_is_macos_true_when_platform_is_darwin():
    with patch("ochat_calendar.platform.system", return_value="Darwin"):
        assert ochat_calendar.is_macos() is True


def test_is_macos_false_when_platform_is_not_darwin():
    with patch("ochat_calendar.platform.system", return_value="Linux"):
        assert ochat_calendar.is_macos() is False
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run --with pytest --with numpy --with requests pytest tests/test_calendar.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'ochat_calendar'`

- [ ] **Step 3: Write minimal implementation**

Create `ochat_calendar.py`:

```python
"""ochat_calendar: macOS Calendar.app I/O via AppleScript (osascript).

No model calls, no prompts, no CLI -- a pure I/O layer that ochat.py's
orchestration code decides when and why to call.
"""

import platform
import subprocess
from datetime import datetime


class CalendarError(Exception):
    """Raised when an osascript call to Calendar.app fails."""


def is_macos() -> bool:
    return platform.system() == "Darwin"
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run --with pytest --with numpy --with requests pytest tests/test_calendar.py -v`
Expected: PASS (8 tests total)

- [ ] **Step 5: Commit**

```bash
git add ochat_calendar.py tests/test_calendar.py
git commit -m "feat: add ochat_calendar module skeleton"
```

---

### Task 4: `_run_applescript()` subprocess helper

**Files:**
- Modify: `ochat_calendar.py`
- Test: `tests/test_calendar.py`

**Interfaces:**
- Consumes: `ochat_calendar.CalendarError`
- Produces: `ochat_calendar._run_applescript(script: str, timeout: float) -> str`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_calendar.py`:

```python
from unittest.mock import MagicMock
import subprocess as subprocess_module


def test_run_applescript_returns_stdout_on_success():
    fake_result = MagicMock(returncode=0, stdout="hello\n", stderr="")
    with patch("ochat_calendar.subprocess.run", return_value=fake_result) as mock_run:
        result = ochat_calendar._run_applescript("return \"hello\"", timeout=5)
    assert result == "hello\n"
    mock_run.assert_called_once()
    assert mock_run.call_args.args[0] == ["osascript", "-e", "return \"hello\""]


def test_run_applescript_raises_calendar_error_on_nonzero_exit():
    fake_result = MagicMock(returncode=1, stdout="", stderr="some AppleScript error")
    with patch("ochat_calendar.subprocess.run", return_value=fake_result):
        try:
            ochat_calendar._run_applescript("bad script", timeout=5)
            assert False, "expected CalendarError"
        except ochat_calendar.CalendarError as exc:
            assert "some AppleScript error" in str(exc)


def test_run_applescript_raises_calendar_error_on_timeout():
    with patch(
        "ochat_calendar.subprocess.run",
        side_effect=subprocess_module.TimeoutExpired(cmd="osascript", timeout=5),
    ):
        try:
            ochat_calendar._run_applescript("slow script", timeout=5)
            assert False, "expected CalendarError"
        except ochat_calendar.CalendarError:
            pass


def test_run_applescript_raises_calendar_error_when_osascript_missing():
    with patch("ochat_calendar.subprocess.run", side_effect=FileNotFoundError("no osascript")):
        try:
            ochat_calendar._run_applescript("script", timeout=5)
            assert False, "expected CalendarError"
        except ochat_calendar.CalendarError:
            pass
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run --with pytest --with numpy --with requests pytest tests/test_calendar.py -v`
Expected: FAIL with `AttributeError: module 'ochat_calendar' has no attribute '_run_applescript'`

- [ ] **Step 3: Write minimal implementation**

In `ochat_calendar.py`, after `is_macos`:

```python
def _run_applescript(script: str, timeout: float) -> str:
    try:
        result = subprocess.run(
            ["osascript", "-e", script],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except (subprocess.TimeoutExpired, FileNotFoundError) as exc:
        raise CalendarError(f"osascript call failed: {exc}") from exc
    if result.returncode != 0:
        raise CalendarError(f"osascript exited {result.returncode}: {result.stderr.strip()}")
    return result.stdout
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run --with pytest --with numpy --with requests pytest tests/test_calendar.py -v`
Expected: PASS (12 tests total)

- [ ] **Step 5: Commit**

```bash
git add ochat_calendar.py tests/test_calendar.py
git commit -m "feat: add AppleScript subprocess runner with error handling"
```

---

### Task 5: `fetch_upcoming_events()` — script builder + output parser

**Files:**
- Modify: `ochat_calendar.py`
- Test: `tests/test_calendar.py`

**Interfaces:**
- Consumes: `ochat_calendar._run_applescript`
- Produces: `ochat_calendar.fetch_upcoming_events(days_ahead: int, timeout: float) -> list[dict]` where each dict is `{"title": str, "start": iso str, "end": iso str, "calendar": str}`; `ochat_calendar._FIELD_SEP`, `ochat_calendar._RECORD_SEP` (module constants, `chr(31)`/`chr(30)`)

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_calendar.py`:

```python
def test_build_fetch_script_includes_days_ahead_and_separators():
    script = ochat_calendar._build_fetch_script(7)
    assert "7 * days" in script
    assert "ASCII character 31" in script
    assert "ASCII character 30" in script
    assert "every event" in script


def test_parse_events_extracts_fields_from_delimited_output():
    sep = ochat_calendar._FIELD_SEP
    rec = ochat_calendar._RECORD_SEP
    raw = (
        f"Dentist{sep}2026{sep}6{sep}25{sep}14{sep}0"
        f"{sep}2026{sep}6{sep}25{sep}14{sep}30{sep}Home{rec}"
    )
    events = ochat_calendar._parse_events(raw)
    assert events == [
        {
            "title": "Dentist",
            "start": "2026-06-25T14:00:00",
            "end": "2026-06-25T14:30:00",
            "calendar": "Home",
        }
    ]


def test_parse_events_skips_malformed_records():
    raw = f"incomplete record{ochat_calendar._RECORD_SEP}"
    assert ochat_calendar._parse_events(raw) == []


def test_parse_events_handles_empty_input():
    assert ochat_calendar._parse_events("") == []


def test_fetch_upcoming_events_returns_parsed_events():
    sep = ochat_calendar._FIELD_SEP
    rec = ochat_calendar._RECORD_SEP
    raw = f"Standup{sep}2026{sep}6{sep}21{sep}9{sep}0{sep}2026{sep}6{sep}21{sep}9{sep}15{sep}Work{rec}"
    with patch("ochat_calendar._run_applescript", return_value=raw) as mock_run:
        events = ochat_calendar.fetch_upcoming_events(7, timeout=10)
    assert len(events) == 1
    assert events[0]["title"] == "Standup"
    mock_run.assert_called_once()
    assert mock_run.call_args.args[1] == 10
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run --with pytest --with numpy --with requests pytest tests/test_calendar.py -v`
Expected: FAIL with `AttributeError: module 'ochat_calendar' has no attribute '_build_fetch_script'`

- [ ] **Step 3: Write minimal implementation**

In `ochat_calendar.py`, after the imports, add the separator constants; after `_run_applescript`, add the builder/parser/public function:

```python
_FIELD_SEP = chr(31)
_RECORD_SEP = chr(30)


def _build_fetch_script(days_ahead: int) -> str:
    sep = f"(ASCII character 31)"
    rec = f"(ASCII character 30)"
    return (
        "set startDate to current date\n"
        f"set endDate to startDate + ({days_ahead} * days)\n"
        "set output to \"\"\n"
        "tell application \"Calendar\"\n"
        "    repeat with cal in calendars\n"
        "        set theEvents to (every event of cal whose start date is greater than or equal to startDate and start date is less than or equal to endDate)\n"
        "        repeat with evt in theEvents\n"
        f"            set output to output & (summary of evt) & {sep} & (year of (start date of evt)) & {sep} & ((month of (start date of evt)) as integer) & {sep} & (day of (start date of evt)) & {sep} & (hours of (start date of evt)) & {sep} & (minutes of (start date of evt)) & {sep} & (year of (end date of evt)) & {sep} & ((month of (end date of evt)) as integer) & {sep} & (day of (end date of evt)) & {sep} & (hours of (end date of evt)) & {sep} & (minutes of (end date of evt)) & {sep} & (name of cal) & {rec}\n"
        "        end repeat\n"
        "    end repeat\n"
        "end tell\n"
        "return output\n"
    )


def _parse_events(raw: str) -> list[dict]:
    events = []
    for record in raw.split(_RECORD_SEP):
        record = record.strip()
        if not record:
            continue
        fields = record.split(_FIELD_SEP)
        if len(fields) != 12:
            continue
        title, sy, sm, sd, sh, smin, ey, em, ed, eh, emin, calendar = fields
        try:
            start = datetime(int(sy), int(sm), int(sd), int(sh), int(smin))
            end = datetime(int(ey), int(em), int(ed), int(eh), int(emin))
        except ValueError:
            continue
        events.append(
            {
                "title": title,
                "start": start.isoformat(),
                "end": end.isoformat(),
                "calendar": calendar,
            }
        )
    return events


def fetch_upcoming_events(days_ahead: int, timeout: float) -> list[dict]:
    script = _build_fetch_script(days_ahead)
    raw = _run_applescript(script, timeout)
    return _parse_events(raw)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run --with pytest --with numpy --with requests pytest tests/test_calendar.py -v`
Expected: PASS (17 tests total)

- [ ] **Step 5: Commit**

```bash
git add ochat_calendar.py tests/test_calendar.py
git commit -m "feat: add fetch_upcoming_events with AppleScript date-range query"
```

---

### Task 6: `create_event()` — locale-independent date construction + escaping

**Files:**
- Modify: `ochat_calendar.py`
- Test: `tests/test_calendar.py`

**Interfaces:**
- Consumes: `ochat_calendar._run_applescript`
- Produces: `ochat_calendar.create_event(title: str, start: datetime, end: datetime, notes: str | None, timeout: float) -> None`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_calendar.py`:

```python
def test_escape_applescript_string_escapes_quotes_and_backslashes():
    assert ochat_calendar._escape_applescript_string('say "hi" \\ done') == 'say \\"hi\\" \\\\ done'


def test_format_applescript_date_setup_sets_each_field():
    when = datetime(2026, 6, 25, 14, 30)
    script = ochat_calendar._format_applescript_date_setup("startDate", when)
    assert "set year of startDate to 2026" in script
    assert "set month of startDate to 6" in script
    assert "set day of startDate to 25" in script
    assert "set hours of startDate to 14" in script
    assert "set minutes of startDate to 30" in script


def test_create_event_runs_applescript_with_title_and_dates():
    start = datetime(2026, 6, 25, 14, 0)
    end = datetime(2026, 6, 25, 14, 30)
    with patch("ochat_calendar._run_applescript", return_value="") as mock_run:
        ochat_calendar.create_event("Dentist", start, end, notes=None, timeout=10)
    script = mock_run.call_args.args[0]
    assert "Dentist" in script
    assert "set year of startDate to 2026" in script
    assert "set hours of endDate to 14" in script
    assert "make new event" in script
    assert mock_run.call_args.args[1] == 10


def test_create_event_escapes_quotes_in_title():
    start = datetime(2026, 6, 25, 14, 0)
    end = datetime(2026, 6, 25, 14, 30)
    with patch("ochat_calendar._run_applescript", return_value="") as mock_run:
        ochat_calendar.create_event('Say "hi"', start, end, notes=None, timeout=10)
    script = mock_run.call_args.args[0]
    assert 'Say \\"hi\\"' in script


def test_create_event_raises_calendar_error_on_applescript_failure():
    start = datetime(2026, 6, 25, 14, 0)
    end = datetime(2026, 6, 25, 14, 30)
    with patch("ochat_calendar._run_applescript", side_effect=ochat_calendar.CalendarError("denied")):
        try:
            ochat_calendar.create_event("Dentist", start, end, notes=None, timeout=10)
            assert False, "expected CalendarError"
        except ochat_calendar.CalendarError:
            pass
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run --with pytest --with numpy --with requests pytest tests/test_calendar.py -v`
Expected: FAIL with `AttributeError: module 'ochat_calendar' has no attribute '_escape_applescript_string'`

- [ ] **Step 3: Write minimal implementation**

In `ochat_calendar.py`, after `fetch_upcoming_events`:

```python
def _escape_applescript_string(value: str) -> str:
    return value.replace("\\", "\\\\").replace('"', '\\"')


def _format_applescript_date_setup(var_name: str, when: datetime) -> str:
    return (
        f"set day of {var_name} to 1\n"
        f"set year of {var_name} to {when.year}\n"
        f"set month of {var_name} to {when.month}\n"
        f"set day of {var_name} to {when.day}\n"
        f"set hours of {var_name} to {when.hour}\n"
        f"set minutes of {var_name} to {when.minute}\n"
        f"set seconds of {var_name} to 0\n"
    )


def create_event(title: str, start: datetime, end: datetime, notes: str | None, timeout: float) -> None:
    safe_title = _escape_applescript_string(title)
    safe_notes = _escape_applescript_string(notes or "")
    script = (
        "set startDate to current date\n"
        + _format_applescript_date_setup("startDate", start)
        + "set endDate to current date\n"
        + _format_applescript_date_setup("endDate", end)
        + "tell application \"Calendar\"\n"
        + "    tell calendar 1\n"
        + f'        make new event at end with properties {{summary:"{safe_title}", start date:startDate, end date:endDate, description:"{safe_notes}"}}\n'
        + "    end tell\n"
        + "end tell\n"
    )
    _run_applescript(script, timeout)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run --with pytest --with numpy --with requests pytest tests/test_calendar.py -v`
Expected: PASS (22 tests total)

- [ ] **Step 5: Commit**

```bash
git add ochat_calendar.py tests/test_calendar.py
git commit -m "feat: add create_event with locale-independent date construction"
```

---

### Task 7: Generalize `_extract_json_array` into a shared array/object helper

**Files:**
- Modify: `ochat.py:264-282`
- Test: `tests/test_calendar.py`

**Interfaces:**
- Produces: `ochat._extract_json_substring(text: str, open_char: str, close_char: str) -> str`, `ochat._extract_json_object(text: str) -> str`
- Note: `ochat._extract_json_array(text: str) -> str` keeps its exact existing behavior (all of `tests/test_memory.py`'s existing fenced/prose-JSON tests must keep passing unchanged)

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_calendar.py`:

```python
def test_extract_json_object_strips_markdown_fence():
    text = '```json\n{"intent": "create"}\n```'
    assert ochat._extract_json_object(text) == '{"intent": "create"}'


def test_extract_json_object_handles_prose_wrapped_json():
    text = 'Sure, here it is: {"intent": "query"} -- done'
    assert ochat._extract_json_object(text) == '{"intent": "query"}'


def test_extract_json_array_unchanged_behavior():
    text = '```json\n["fact one"]\n```'
    assert ochat._extract_json_array(text) == '["fact one"]'
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run --with pytest --with numpy --with requests pytest tests/test_calendar.py -v`
Expected: FAIL with `AttributeError: module 'ochat' has no attribute '_extract_json_object'`

- [ ] **Step 3: Write minimal implementation**

Replace `ochat.py:264-282` (the existing `_extract_json_array` function):

```python
def _extract_json_substring(text: str, open_char: str, close_char: str) -> str:
    """Pull a clean JSON substring out of model output.

    Models sometimes wrap their JSON reply in markdown code fences or
    surround it with prose. Strip fences if present, then fall back to
    slicing between the first opening char and the last closing char so
    json.loads has a fighting chance.
    """
    text = text.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.lower().startswith("json"):
            text = text[4:]
        text = text.strip()
    start = text.find(open_char)
    end = text.rfind(close_char)
    if start != -1 and end != -1 and end > start:
        return text[start : end + 1]
    return text


def _extract_json_array(text: str) -> str:
    return _extract_json_substring(text, "[", "]")


def _extract_json_object(text: str) -> str:
    return _extract_json_substring(text, "{", "}")
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run --with pytest --with numpy --with requests pytest tests/test_calendar.py tests/test_memory.py -v`
Expected: PASS (all existing `test_memory.py` tests plus the 3 new ones, 0 regressions)

- [ ] **Step 5: Commit**

```bash
git add ochat.py tests/test_calendar.py
git commit -m "refactor: generalize JSON-substring extraction for arrays and objects"
```

---

### Task 8: `CALENDAR_INTENT_PROMPT` + `classify_calendar_intent()`

**Files:**
- Modify: `ochat.py` (add after `_extract_json_object`, before `extract_facts`)
- Test: `tests/test_calendar.py`

**Interfaces:**
- Consumes: `ochat.ollama_chat`, `ochat._extract_json_object`
- Produces: `ochat.CALENDAR_INTENT_PROMPT`, `ochat.classify_calendar_intent(user_input: str, now_context: str) -> dict` returning `{"intent": "none"|"query"|"create", "title": str|None, "start": str|None, "end": str|None, "notes": str|None}`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_calendar.py`:

```python
def test_classify_calendar_intent_parses_create_response():
    response = (
        '{"intent": "create", "title": "Dentist", '
        '"start": "2026-06-25T14:00:00", "end": "2026-06-25T14:30:00", "notes": null}'
    )
    with patch("ochat.ollama_chat", return_value=response):
        result = ochat.classify_calendar_intent("add a dentist appt thursday 2pm", "Current date/time: ...")
    assert result == {
        "intent": "create",
        "title": "Dentist",
        "start": "2026-06-25T14:00:00",
        "end": "2026-06-25T14:30:00",
        "notes": None,
    }


def test_classify_calendar_intent_parses_query_response():
    response = '{"intent": "query", "title": null, "start": null, "end": null, "notes": null}'
    with patch("ochat.ollama_chat", return_value=response):
        result = ochat.classify_calendar_intent("what's on my calendar friday", "Current date/time: ...")
    assert result["intent"] == "query"


def test_classify_calendar_intent_handles_fenced_json():
    response = '```json\n{"intent": "none", "title": null, "start": null, "end": null, "notes": null}\n```'
    with patch("ochat.ollama_chat", return_value=response):
        result = ochat.classify_calendar_intent("how's the weather", "Current date/time: ...")
    assert result["intent"] == "none"


def test_classify_calendar_intent_falls_back_to_none_on_malformed_json():
    with patch("ochat.ollama_chat", return_value="not json at all"):
        result = ochat.classify_calendar_intent("garbage in", "Current date/time: ...")
    assert result["intent"] == "none"


def test_classify_calendar_intent_falls_back_to_none_on_model_failure():
    with patch("ochat.ollama_chat", side_effect=RuntimeError("model unreachable")):
        result = ochat.classify_calendar_intent("add a meeting", "Current date/time: ...")
    assert result["intent"] == "none"


def test_classify_calendar_intent_passes_now_context_to_system_prompt():
    response = '{"intent": "none", "title": null, "start": null, "end": null, "notes": null}'
    with patch("ochat.ollama_chat", return_value=response) as mock_chat:
        ochat.classify_calendar_intent("hi", "Current date/time: Saturday, June 20, 2026")
    system_message = mock_chat.call_args.args[0][0]["content"]
    assert "Current date/time: Saturday, June 20, 2026" in system_message
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run --with pytest --with numpy --with requests pytest tests/test_calendar.py -v`
Expected: FAIL with `AttributeError: module 'ochat' has no attribute 'classify_calendar_intent'`

- [ ] **Step 3: Write minimal implementation**

In `ochat.py`, after `_extract_json_object`:

```python
CALENDAR_INTENT_PROMPT = (
    "You help detect calendar-related requests. Given the current date/time "
    "context and a user message, respond with ONLY a JSON object describing "
    "the user's calendar intent: "
    '{"intent": "none"|"query"|"create", "title": string or null, '
    '"start": string or null, "end": string or null, "notes": string or null}. '
    'Use intent "create" only if the user is clearly asking to add a new '
    "event/appointment/meeting to their calendar. Use intent \"query\" if "
    "they're asking what's on their calendar or about existing events. "
    "Otherwise use \"none\". When intent is \"create\", resolve any relative "
    'dates/times (e.g. "next Thursday", "tomorrow at 2pm") to absolute ISO '
    '8601 datetimes (YYYY-MM-DDTHH:MM:SS) for "start" and "end" using the '
    "current date/time context given. If no explicit end time is mentioned, "
    "assume the event is 1 hour long."
)


def classify_calendar_intent(user_input: str, now_context: str) -> dict:
    fallback = {"intent": "none", "title": None, "start": None, "end": None, "notes": None}
    try:
        reply = ollama_chat(
            [
                {"role": "system", "content": f"{CALENDAR_INTENT_PROMPT}\n\n{now_context}"},
                {"role": "user", "content": user_input},
            ],
            think=False,
            stream_to_stdout=False,
        )
        parsed = json.loads(_extract_json_object(reply))
        if not isinstance(parsed, dict) or parsed.get("intent") not in ("none", "query", "create"):
            return fallback
        return {
            "intent": parsed.get("intent"),
            "title": parsed.get("title"),
            "start": parsed.get("start"),
            "end": parsed.get("end"),
            "notes": parsed.get("notes"),
        }
    except Exception:
        return fallback
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run --with pytest --with numpy --with requests pytest tests/test_calendar.py -v`
Expected: PASS (31 tests total)

- [ ] **Step 5: Commit**

```bash
git add ochat.py tests/test_calendar.py
git commit -m "feat: add calendar intent classification via ollama_chat"
```

---

### Task 9: `build_system_prompt()` — datetime context + calendar events section

**Files:**
- Modify: `ochat.py:312-317`
- Test: `tests/test_calendar.py`

**Interfaces:**
- Consumes: `ochat.current_datetime_context`
- Produces: `ochat.build_system_prompt(relevant_facts, calendar_events=None) -> str`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_calendar.py`:

```python
def test_build_system_prompt_includes_current_datetime_context():
    prompt = ochat.build_system_prompt([])
    assert "Current date/time:" in prompt


def test_build_system_prompt_includes_calendar_events_section():
    events = [{"title": "Standup", "start": "2026-06-21T09:00:00", "end": "2026-06-21T09:15:00", "calendar": "Work"}]
    prompt = ochat.build_system_prompt([], calendar_events=events)
    assert "Upcoming calendar events" in prompt
    assert "Standup" in prompt


def test_build_system_prompt_omits_calendar_section_when_no_events():
    prompt = ochat.build_system_prompt([], calendar_events=[])
    assert "Upcoming calendar events" not in prompt
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run --with pytest --with numpy --with requests pytest tests/test_calendar.py -v`
Expected: FAIL — `test_build_system_prompt_includes_current_datetime_context` fails because the base prompt doesn't yet include the datetime line.

- [ ] **Step 3: Write minimal implementation**

Replace `ochat.py:312-317`:

```python
def build_system_prompt(relevant_facts, calendar_events=None):
    sections = [
        f"You are a helpful assistant talking with the user in their terminal.\n\n{current_datetime_context()}"
    ]
    if relevant_facts:
        bullets = "\n".join(f"- {fact['text']}" for fact in relevant_facts)
        sections.append(f"Relevant memory:\n{bullets}")
    if calendar_events:
        bullets = "\n".join(
            f"- {event['title']} ({event['start']} to {event['end']}, {event['calendar']})"
            for event in calendar_events
        )
        sections.append(f"Upcoming calendar events:\n{bullets}")
    return "\n\n".join(sections)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run --with pytest --with numpy --with requests pytest tests/test_calendar.py tests/test_memory.py -v`
Expected: PASS — including the existing `test_build_system_prompt_with_no_facts` and `test_build_system_prompt_includes_fact_bullets` in `test_memory.py`, which only assert substrings and are unaffected by the new datetime line.

- [ ] **Step 5: Commit**

```bash
git add ochat.py tests/test_calendar.py
git commit -m "feat: include current datetime and calendar events in system prompt"
```

---

### Task 10: `EXTRACTION_PROMPT` + `extract_facts()` — date-resolution instruction

**Files:**
- Modify: `ochat.py:250-255` (`EXTRACTION_PROMPT`), `ochat.py:285-309` (`extract_facts`)
- Test: `tests/test_calendar.py`

**Interfaces:**
- Consumes: `ochat.current_datetime_context`
- Produces: updated `ochat.EXTRACTION_PROMPT` text; `extract_facts` unchanged signature

- [ ] **Step 1: Write the failing test**

Append to `tests/test_calendar.py`:

```python
def test_extract_facts_includes_current_datetime_in_system_prompt(tmp_path):
    conn = ochat.init_db(tmp_path / "memory.db")
    with patch("ochat.EXTRACTION_LOG_PATH", tmp_path / "extraction.log"), \
         patch("ochat.ollama_chat", return_value="[]") as mock_chat, \
         patch("ochat.ollama_embed", return_value=__import__("numpy").array([1.0], dtype="float32")):
        ochat.extract_facts(conn, "let's meet next Thursday", "sounds good", "default")
    system_message = mock_chat.call_args.args[0][0]["content"]
    assert "Current date/time:" in system_message
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run --with pytest --with numpy --with requests pytest tests/test_calendar.py -v`
Expected: FAIL — `assert "Current date/time:" in system_message` fails because the extraction system prompt doesn't include it yet.

- [ ] **Step 3: Write minimal implementation**

Replace `ochat.py:250-255`:

```python
EXTRACTION_PROMPT = (
    "Given this exchange, list any new durable facts or preferences about the "
    "user worth remembering long-term. Respond with ONLY a JSON array of short "
    'fact strings, e.g. ["prefers terse answers"]. If nothing is worth '
    "remembering, respond with []. Resolve any relative dates mentioned (e.g. "
    "'next Thursday') to absolute dates before recording a fact, using the "
    "current date/time context given."
)
```

In `ochat.py:285-309` (`extract_facts`), change the system message content from `EXTRACTION_PROMPT` to include the datetime context:

```python
        reply = ollama_chat(
            [
                {"role": "system", "content": f"{EXTRACTION_PROMPT}\n\n{current_datetime_context()}"},
                {"role": "user", "content": exchange},
            ],
            think=False,
            stream_to_stdout=False,
        )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run --with pytest --with numpy --with requests pytest tests/test_calendar.py tests/test_memory.py -v`
Expected: PASS — all existing `extract_facts` tests in `test_memory.py` still pass unchanged (they assert on resulting fact rows, not exact prompt text).

- [ ] **Step 5: Commit**

```bash
git add ochat.py tests/test_calendar.py
git commit -m "feat: resolve relative dates during fact extraction via datetime context"
```

---

### Task 11: Calendar constants, `refresh_calendar_cache()`, `handle_calendar_create_intent()`, and `handle_turn()` integration

**Files:**
- Modify: `ochat.py` (top-level import + constants near ochat.py:14-35; new functions before `handle_turn`; `handle_turn` itself at ochat.py:320-354)
- Test: `tests/test_calendar.py`

**Interfaces:**
- Consumes: `ochat_calendar.is_macos`, `ochat_calendar.fetch_upcoming_events`, `ochat_calendar.create_event`, `ochat_calendar.CalendarError`, `ochat.classify_calendar_intent`, `ochat.looks_calendar_related`, `ochat.current_datetime_context`, `ochat.build_system_prompt`
- Produces: `ochat.refresh_calendar_cache(calendar_cache: dict) -> list[dict]`, `ochat.handle_calendar_create_intent(user_input: str, now_context: str, calendar_cache: dict) -> None`, `ochat.handle_turn(conn, thread, path, user_input, think, calendar_cache=None)` (new optional 6th parameter; existing 5-arg call sites in `tests/test_memory.py` continue to work unchanged since `calendar_cache` defaults to `None` and the whole calendar block is skipped when `None`)

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_calendar.py`:

```python
def test_refresh_calendar_cache_fetches_when_empty():
    events = [{"title": "Standup", "start": "2026-06-21T09:00:00", "end": "2026-06-21T09:15:00", "calendar": "Work"}]
    cache = {"events": [], "fetched_at": None}
    with patch("ochat.ochat_calendar.fetch_upcoming_events", return_value=events) as mock_fetch:
        result = ochat.refresh_calendar_cache(cache)
    assert result == events
    assert cache["events"] == events
    mock_fetch.assert_called_once()


def test_refresh_calendar_cache_reuses_fresh_cache():
    cache = {"events": [{"title": "cached"}], "fetched_at": datetime.now(timezone.utc)}
    with patch("ochat.ochat_calendar.fetch_upcoming_events") as mock_fetch:
        result = ochat.refresh_calendar_cache(cache)
    assert result == [{"title": "cached"}]
    mock_fetch.assert_not_called()


def test_refresh_calendar_cache_keeps_last_known_on_error():
    cache = {"events": [{"title": "stale"}], "fetched_at": None}
    with patch("ochat.ochat_calendar.fetch_upcoming_events", side_effect=ochat.ochat_calendar.CalendarError("denied")):
        result = ochat.refresh_calendar_cache(cache)
    assert result == [{"title": "stale"}]


def test_handle_calendar_create_intent_confirmed_calls_create_event():
    cache = {"events": [], "fetched_at": None}
    intent_response = (
        '{"intent": "create", "title": "Dentist", "start": "2026-06-25T14:00:00", '
        '"end": "2026-06-25T14:30:00", "notes": null}'
    )
    with patch("ochat.ollama_chat", return_value=intent_response), \
         patch("builtins.input", return_value="y"), \
         patch("ochat.ochat_calendar.create_event") as mock_create:
        ochat.handle_calendar_create_intent("add a dentist appt thursday 2pm", "Current date/time: ...", cache)
    mock_create.assert_called_once()
    assert cache["events"][0]["title"] == "Dentist"


def test_handle_calendar_create_intent_declined_skips_create_event():
    cache = {"events": [], "fetched_at": None}
    intent_response = (
        '{"intent": "create", "title": "Dentist", "start": "2026-06-25T14:00:00", '
        '"end": "2026-06-25T14:30:00", "notes": null}'
    )
    with patch("ochat.ollama_chat", return_value=intent_response), \
         patch("builtins.input", return_value="n"), \
         patch("ochat.ochat_calendar.create_event") as mock_create:
        ochat.handle_calendar_create_intent("add a dentist appt thursday 2pm", "Current date/time: ...", cache)
    mock_create.assert_not_called()
    assert cache["events"] == []


def test_handle_calendar_create_intent_query_does_nothing():
    cache = {"events": [], "fetched_at": None}
    with patch("ochat.ollama_chat", return_value='{"intent": "query", "title": null, "start": null, "end": null, "notes": null}'), \
         patch("ochat.ochat_calendar.create_event") as mock_create:
        ochat.handle_calendar_create_intent("what's on my calendar friday", "Current date/time: ...", cache)
    mock_create.assert_not_called()


def test_handle_calendar_create_intent_create_failure_does_not_raise():
    cache = {"events": [], "fetched_at": None}
    intent_response = (
        '{"intent": "create", "title": "Dentist", "start": "2026-06-25T14:00:00", '
        '"end": "2026-06-25T14:30:00", "notes": null}'
    )
    with patch("ochat.ollama_chat", return_value=intent_response), \
         patch("builtins.input", return_value="y"), \
         patch("ochat.ochat_calendar.create_event", side_effect=ochat.ochat_calendar.CalendarError("denied")):
        ochat.handle_calendar_create_intent("add a dentist appt thursday 2pm", "Current date/time: ...", cache)  # must not raise
    assert cache["events"] == []


def test_handle_turn_skips_calendar_block_when_cache_is_none(tmp_path):
    conn = ochat.init_db(tmp_path / "memory.db")
    thread = {"name": "t", "messages": []}
    path = tmp_path / "t.json"
    with patch("ochat.ollama_embed", return_value=__import__("numpy").array([1.0], dtype="float32")), \
         patch("ochat.ollama_chat", return_value="hi there"), \
         patch("ochat.extract_facts"):
        result = ochat.handle_turn(conn, thread, path, "add a meeting thursday", "off")
    assert result is not None
    result.join(timeout=2)


def test_handle_turn_includes_calendar_events_in_prompt_when_cache_given(tmp_path):
    conn = ochat.init_db(tmp_path / "memory.db")
    thread = {"name": "t", "messages": []}
    path = tmp_path / "t.json"
    cache = {"events": [], "fetched_at": None}
    events = [{"title": "Standup", "start": "2026-06-21T09:00:00", "end": "2026-06-21T09:15:00", "calendar": "Work"}]
    captured_payloads = []

    def fake_chat(messages, think=False, stream_to_stdout=True):
        captured_payloads.append(messages)
        return "hi there"

    with patch("ochat.ochat_calendar.is_macos", return_value=True), \
         patch("ochat.ochat_calendar.fetch_upcoming_events", return_value=events), \
         patch("ochat.ollama_embed", return_value=__import__("numpy").array([1.0], dtype="float32")), \
         patch("ochat.ollama_chat", side_effect=fake_chat), \
         patch("ochat.extract_facts"):
        result = ochat.handle_turn(conn, thread, path, "what's the weather", "off", cache)
    result.join(timeout=2)
    system_message = captured_payloads[0][0]["content"]
    assert "Standup" in system_message
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run --with pytest --with numpy --with requests pytest tests/test_calendar.py -v`
Expected: FAIL with `AttributeError: module 'ochat' has no attribute 'refresh_calendar_cache'`

- [ ] **Step 3: Write minimal implementation**

In `ochat.py`, add the import next to the existing imports (after `import threading` near line 13):

```python
import ochat_calendar
```

Add the new constants after `EXTRACTION_JOIN_TIMEOUT_SECONDS = 5` (line 35):

```python
CALENDAR_ENABLED = True
CALENDAR_LOOKAHEAD_DAYS = 7
CALENDAR_CACHE_TTL_SECONDS = 300
APPLESCRIPT_TIMEOUT_SECONDS = 10
```

Add the two new orchestration functions immediately before `handle_turn` (before line 320):

```python
def refresh_calendar_cache(calendar_cache: dict) -> list[dict]:
    fetched_at = calendar_cache.get("fetched_at")
    if fetched_at is not None and (datetime.now(timezone.utc) - fetched_at).total_seconds() < CALENDAR_CACHE_TTL_SECONDS:
        return calendar_cache.get("events", [])
    try:
        events = ochat_calendar.fetch_upcoming_events(CALENDAR_LOOKAHEAD_DAYS, APPLESCRIPT_TIMEOUT_SECONDS)
        calendar_cache["events"] = events
        calendar_cache["fetched_at"] = datetime.now(timezone.utc)
    except ochat_calendar.CalendarError as exc:
        print(f"\nwarning: calendar read failed ({exc}); continuing with last-known events", file=sys.stderr)
    return calendar_cache.get("events", [])


def handle_calendar_create_intent(user_input: str, now_context: str, calendar_cache: dict) -> None:
    intent_result = classify_calendar_intent(user_input, now_context)
    if intent_result["intent"] != "create":
        return
    title = intent_result.get("title")
    start_raw = intent_result.get("start")
    end_raw = intent_result.get("end")
    if not title or not start_raw or not end_raw:
        return
    try:
        start = datetime.fromisoformat(start_raw)
        end = datetime.fromisoformat(end_raw)
    except ValueError:
        return
    confirm_text = (
        f'Add to calendar? "{title}" -- '
        f'{start.strftime("%a, %b %d %Y, %I:%M %p")}-{end.strftime("%I:%M %p")} [y/N] '
    )
    answer = input(confirm_text).strip().lower()
    if answer not in ("y", "yes"):
        print("cancelled, nothing was added")
        return
    try:
        ochat_calendar.create_event(title, start, end, intent_result.get("notes"), APPLESCRIPT_TIMEOUT_SECONDS)
    except ochat_calendar.CalendarError as exc:
        print(f"error: failed to add event ({exc})", file=sys.stderr)
        return
    calendar_cache["events"] = calendar_cache.get("events", []) + [
        {"title": title, "start": start.isoformat(), "end": end.isoformat(), "calendar": ""}
    ]
```

Replace `handle_turn` (`ochat.py:320-354`) with:

```python
def handle_turn(conn, thread, path, user_input, think, calendar_cache=None):
    try:
        query_embedding = ollama_embed(user_input)
    except requests.RequestException as exc:
        print(f"\nerror: chat request failed ({exc}); message not saved, try again", file=sys.stderr)
        return None

    try:
        relevant = top_k_facts(query_embedding, get_all_facts(conn))
    except Exception as exc:
        print(f"\nwarning: fact retrieval failed ({exc}); continuing without relevant facts", file=sys.stderr)
        relevant = []

    calendar_events = []
    if calendar_cache is not None and CALENDAR_ENABLED and ochat_calendar.is_macos():
        now_context = current_datetime_context()
        calendar_events = refresh_calendar_cache(calendar_cache)
        if looks_calendar_related(user_input):
            handle_calendar_create_intent(user_input, now_context, calendar_cache)
            calendar_events = calendar_cache.get("events", calendar_events)

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
    now = datetime.now(timezone.utc).isoformat()
    thread["messages"].append({"role": "user", "content": user_input, "ts": now})
    thread["messages"].append({"role": "assistant", "content": reply, "ts": now})
    save_thread(path, thread)
    extraction_thread = threading.Thread(
        target=extract_facts,
        args=(conn, user_input, reply, thread["name"]),
        daemon=True,
    )
    extraction_thread.start()
    return extraction_thread
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run --with pytest --with numpy --with requests pytest tests/test_calendar.py tests/test_memory.py -v`
Expected: PASS — all tests in both files, including every pre-existing `handle_turn` test in `test_memory.py` (which calls `handle_turn` with 5 positional args and relies on `calendar_cache` defaulting to `None`).

- [ ] **Step 5: Commit**

```bash
git add ochat.py tests/test_calendar.py
git commit -m "feat: wire calendar read/write into handle_turn with cache and confirmation"
```

---

### Task 12: `run_chat_loop()` — create and thread the calendar cache

**Files:**
- Modify: `ochat.py:357-377`

**Interfaces:**
- Consumes: `ochat.handle_turn(..., calendar_cache=None)`
- Produces: no new public interface; `run_chat_loop` has no existing unit tests in `tests/test_memory.py`, so none are added here either, consistent with that file's existing convention of not unit-testing the interactive loop itself.

- [ ] **Step 1: Modify `run_chat_loop`**

Replace `ochat.py:357-377`:

```python
def run_chat_loop(thread_name: str, think: str) -> None:
    check_ollama_ready()
    conn = init_db(MEMORY_DB_PATH)
    path = thread_path(thread_name)
    thread = load_thread(path, thread_name)
    calendar_cache = {"events": [], "fetched_at": None}
    pending_extraction = None
    try:
        while True:
            try:
                user_input = input("you> ").strip()
            except EOFError:
                print()
                break
            if not user_input:
                continue
            if pending_extraction is not None:
                pending_extraction.join(timeout=0)
            pending_extraction = handle_turn(conn, thread, path, user_input, think, calendar_cache)
    finally:
        if pending_extraction is not None:
            pending_extraction.join(timeout=EXTRACTION_JOIN_TIMEOUT_SECONDS)
```

- [ ] **Step 2: Run the full suite to confirm no regressions**

Run: `uv run --with pytest --with numpy --with requests pytest tests/ -v`
Expected: PASS — every test in both `test_memory.py` and `test_calendar.py`

- [ ] **Step 3: Commit**

```bash
git add ochat.py
git commit -m "feat: thread a per-process calendar cache through run_chat_loop"
```

---

### Task 13: `ochat calendar list` CLI subcommand

**Files:**
- Modify: `ochat.py` (new `cmd_calendar_list` before `main`, around ochat.py:400-409; `main`'s argparse setup, `ochat.py:409-433`)
- Test: `tests/test_calendar.py`

**Interfaces:**
- Consumes: `ochat_calendar.is_macos`, `ochat_calendar.fetch_upcoming_events`, `ochat_calendar.CalendarError`
- Produces: `ochat.cmd_calendar_list() -> None`; `main()` gains a `calendar` subcommand with a `list` sub-subcommand

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_calendar.py`:

```python
def test_cmd_calendar_list_prints_events(capsys):
    events = [{"title": "Standup", "start": "2026-06-21T09:00:00", "end": "2026-06-21T09:15:00", "calendar": "Work"}]
    with patch("ochat.ochat_calendar.is_macos", return_value=True), \
         patch("ochat.ochat_calendar.fetch_upcoming_events", return_value=events):
        ochat.cmd_calendar_list()
    out = capsys.readouterr().out
    assert "Standup" in out


def test_cmd_calendar_list_prints_message_when_no_events(capsys):
    with patch("ochat.ochat_calendar.is_macos", return_value=True), \
         patch("ochat.ochat_calendar.fetch_upcoming_events", return_value=[]):
        ochat.cmd_calendar_list()
    assert "no upcoming events" in capsys.readouterr().out


def test_cmd_calendar_list_exits_with_error_off_macos(capsys):
    with patch("ochat.ochat_calendar.is_macos", return_value=False), \
         patch("ochat.sys.exit", side_effect=SystemExit) as mock_exit:
        try:
            ochat.cmd_calendar_list()
        except SystemExit:
            pass
    mock_exit.assert_called_with(1)


def test_cmd_calendar_list_exits_with_error_on_calendar_error(capsys):
    with patch("ochat.ochat_calendar.is_macos", return_value=True), \
         patch("ochat.ochat_calendar.fetch_upcoming_events", side_effect=ochat.ochat_calendar.CalendarError("denied")), \
         patch("ochat.sys.exit", side_effect=SystemExit) as mock_exit:
        try:
            ochat.cmd_calendar_list()
        except SystemExit:
            pass
    mock_exit.assert_called_with(1)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run --with pytest --with numpy --with requests pytest tests/test_calendar.py -v`
Expected: FAIL with `AttributeError: module 'ochat' has no attribute 'cmd_calendar_list'`

- [ ] **Step 3: Write minimal implementation**

In `ochat.py`, add `cmd_calendar_list` after `cmd_memory_forget` (after line 406, before `def main`):

```python
def cmd_calendar_list() -> None:
    if not ochat_calendar.is_macos():
        print("calendar features are only supported on macOS", file=sys.stderr)
        sys.exit(1)
    try:
        events = ochat_calendar.fetch_upcoming_events(CALENDAR_LOOKAHEAD_DAYS, APPLESCRIPT_TIMEOUT_SECONDS)
    except ochat_calendar.CalendarError as exc:
        print(f"error: failed to read calendar ({exc})", file=sys.stderr)
        sys.exit(1)
    if not events:
        print("no upcoming events")
        return
    for event in events:
        print(f"{event['start']} - {event['end']}  {event['title']}  ({event['calendar']})")
```

In `main()` (`ochat.py:409-433`), add a `calendar` subparser after the `memory` subparser block and before `args = parser.parse_args()`:

```python
    calendar_parser = subparsers.add_parser("calendar")
    calendar_sub = calendar_parser.add_subparsers(dest="calendar_command")
    calendar_sub.add_parser("list")
```

And add the dispatch branch alongside the existing `elif args.command == "memory":` branch:

```python
    elif args.command == "calendar":
        if args.calendar_command == "list":
            cmd_calendar_list()
        else:
            calendar_parser.print_help()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run --with pytest --with numpy --with requests pytest tests/test_calendar.py -v`
Expected: PASS (full file)

- [ ] **Step 5: Commit**

```bash
git add ochat.py tests/test_calendar.py
git commit -m "feat: add ochat calendar list CLI subcommand"
```

---

### Task 14: Update documentation

**Files:**
- Modify: `Persememory.md`, `quickuse.md`, `CLAUDE.md`

**Interfaces:** None (documentation only)

- [ ] **Step 1: Update `Persememory.md`**

Add a new `## 16. Clock & Calendar Awareness` section (after the existing `## 15. Repository Layout`) summarizing: `current_datetime_context()` and where it's injected; the `ochat_calendar.py` module and its functions; the read-path cache/TTL; the write-path keyword-gate + intent-classification + confirmation flow; the `ochat calendar list` command; and the macOS-only/graceful-degradation behavior. Base the content on `docs/superpowers/specs/2026-06-20-clock-calendar-design.md`, written in the same reference-table style as the existing sections.

- [ ] **Step 2: Update `quickuse.md`**

Add a `## Calendar awareness` section documenting: that `ochat` now mentions the current date/time to the model automatically; that upcoming events from macOS Calendar.app appear as context (macOS only); that asking to add an event triggers a y/N confirmation before anything is written; the `ochat calendar list` command; and a note that the first calendar access will prompt for a one-time macOS permission approval (System Settings > Privacy & Security > Automation/Calendar) that only the user can grant.

- [ ] **Step 3: Update `CLAUDE.md`**

Add `ochat_calendar.py` to the architecture section's description of the codebase layout and the four-layer split (it's a second I/O-layer file alongside `ochat.py`'s existing "Storage"/"Ollama client" layers), and mention the two new tunables groups (`CALENDAR_*`, `APPLESCRIPT_TIMEOUT_SECONDS`) in the testing/configuration notes.

- [ ] **Step 4: Commit**

```bash
git add Persememory.md quickuse.md CLAUDE.md
git commit -m "docs: document clock and calendar awareness feature"
```

---

### Task 15: Full regression pass and live macOS smoke check

**Files:** None (verification only)

**Interfaces:** None

- [ ] **Step 1: Run the complete automated test suite**

Run: `uv run --with pytest --with numpy --with requests pytest tests/ -v`
Expected: PASS — every test in `tests/test_memory.py` (pre-existing, 0 regressions) and `tests/test_calendar.py` (all new tests from Tasks 1-13).

- [ ] **Step 2: Verify `ochat.py` and `ochat_calendar.py` both import cleanly**

Run: `uv run --with numpy --with requests python3 -c "import ochat; print('ochat OK')"`
Expected: prints `ochat OK` with no errors (confirms `import ochat_calendar` at the top of `ochat.py` doesn't break anything, including on this real macOS machine).

- [ ] **Step 3: Live read-path smoke check against the real Calendar.app on this machine**

Since this development machine is actually macOS, attempt a real (read-only) call:

Run: `uv run --with numpy --with requests python3 -c "import ochat_calendar; print(ochat_calendar.fetch_upcoming_events(7, 10))"`

This either prints a real (possibly empty) list of upcoming events, or raises `CalendarError` if Calendar/Automation permission hasn't been granted yet — both are valid, informative outcomes to report back, not failures of the plan. Do **not** attempt a live `create_event` call as part of this automated step — that writes to the user's real calendar and requires their explicit go-ahead first (see plan notes).

- [ ] **Step 4: Final commit if any fixups were needed**

If Steps 1-3 surfaced any bug fixes, stage and commit them individually with descriptive messages following the same pattern as the tasks above before considering this plan complete.
