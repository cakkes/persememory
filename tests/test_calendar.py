import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import ochat
import ochat_calendar


def test_current_datetime_context_formats_local_time():
    fixed = datetime(2026, 6, 20, 18, 51, tzinfo=timezone(timedelta(hours=-7)))
    with patch("ochat.datetime") as mock_datetime:
        mock_datetime.now.return_value.astimezone.return_value = fixed
        result = ochat.current_datetime_context()
    assert result == "Current date/time: Saturday, June 20, 2026, 06:51 PM UTC-07:00"


def test_looks_calendar_related_true_for_meeting_keyword():
    assert ochat.looks_calendar_related("add a meeting tomorrow at 2pm") is True


def test_looks_calendar_related_true_for_weekday_name():
    assert ochat.looks_calendar_related("let's talk Thursday") is True


def test_looks_calendar_related_is_case_insensitive():
    assert ochat.looks_calendar_related("REMIND me about this") is True


def test_looks_calendar_related_false_for_unrelated_text():
    assert ochat.looks_calendar_related("what's the weather like") is False


def test_calendar_error_is_an_exception():
    assert issubclass(ochat_calendar.CalendarError, Exception)


def test_is_macos_true_when_platform_is_darwin():
    with patch("ochat_calendar.platform.system", return_value="Darwin"):
        assert ochat_calendar.is_macos() is True


def test_is_macos_false_when_platform_is_not_darwin():
    with patch("ochat_calendar.platform.system", return_value="Linux"):
        assert ochat_calendar.is_macos() is False


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
