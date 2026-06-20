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


def test_looks_calendar_related_true_for_meeting_keyword():
    assert ochat.looks_calendar_related("add a meeting tomorrow at 2pm") is True


def test_looks_calendar_related_true_for_weekday_name():
    assert ochat.looks_calendar_related("let's talk Thursday") is True


def test_looks_calendar_related_is_case_insensitive():
    assert ochat.looks_calendar_related("REMIND me about this") is True


def test_looks_calendar_related_false_for_unrelated_text():
    assert ochat.looks_calendar_related("what's the weather like") is False
