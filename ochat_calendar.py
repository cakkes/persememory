"""ochat_calendar: macOS Calendar.app I/O via AppleScript (osascript).

No model calls, no prompts, no CLI -- a pure I/O layer that ochat.py's
orchestration code decides when and why to call.
"""

import platform
import subprocess
from datetime import datetime

_FIELD_SEP = chr(31)
_RECORD_SEP = chr(30)


class CalendarError(Exception):
    """Raised when an osascript call to Calendar.app fails."""


def is_macos() -> bool:
    return platform.system() == "Darwin"


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


def _build_fetch_script(days_ahead: int) -> str:
    sep = "(ASCII character 31)"
    rec = "(ASCII character 30)"
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
