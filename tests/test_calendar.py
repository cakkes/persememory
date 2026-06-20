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


def test_format_applescript_date_setup_sets_fields_in_safe_order():
    when = datetime(2026, 2, 25, 14, 30)
    script = ochat_calendar._format_applescript_date_setup("startDate", when)
    reset_idx = script.index("set day of startDate to 1\n")
    year_idx = script.index("set year of startDate to 2026")
    month_idx = script.index("set month of startDate to 2")
    real_day_idx = script.index("set day of startDate to 25")
    hours_idx = script.index("set hours of startDate to 14")
    minutes_idx = script.index("set minutes of startDate to 30")
    assert reset_idx < year_idx < month_idx < real_day_idx < hours_idx < minutes_idx


def test_create_event_includes_notes_in_script():
    start = datetime(2026, 6, 25, 14, 0)
    end = datetime(2026, 6, 25, 14, 30)
    with patch("ochat_calendar._run_applescript", return_value="") as mock_run:
        ochat_calendar.create_event("Dentist", start, end, notes="Bring insurance card", timeout=10)
    script = mock_run.call_args.args[0]
    assert 'description:"Bring insurance card"' in script


def test_create_event_escapes_quotes_in_notes():
    start = datetime(2026, 6, 25, 14, 0)
    end = datetime(2026, 6, 25, 14, 30)
    with patch("ochat_calendar._run_applescript", return_value="") as mock_run:
        ochat_calendar.create_event("Dentist", start, end, notes='Bring "insurance" card', timeout=10)
    script = mock_run.call_args.args[0]
    assert 'description:"Bring \\"insurance\\" card"' in script


def test_create_event_with_none_notes_produces_empty_description():
    start = datetime(2026, 6, 25, 14, 0)
    end = datetime(2026, 6, 25, 14, 30)
    with patch("ochat_calendar._run_applescript", return_value="") as mock_run:
        ochat_calendar.create_event("Dentist", start, end, notes=None, timeout=10)
    script = mock_run.call_args.args[0]
    assert 'description:""' in script


def test_extract_json_object_strips_markdown_fence():
    text = '```json\n{"intent": "create"}\n```'
    assert ochat._extract_json_object(text) == '{"intent": "create"}'


def test_extract_json_object_handles_prose_wrapped_json():
    text = 'Sure, here it is: {"intent": "query"} -- done'
    assert ochat._extract_json_object(text) == '{"intent": "query"}'


def test_extract_json_array_unchanged_behavior():
    text = '```json\n["fact one"]\n```'
    assert ochat._extract_json_array(text) == '["fact one"]'


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


def test_extract_facts_includes_current_datetime_in_system_prompt(tmp_path):
    conn = ochat.init_db(tmp_path / "memory.db")
    with patch("ochat.EXTRACTION_LOG_PATH", tmp_path / "extraction.log"), \
         patch("ochat.ollama_chat", return_value="[]") as mock_chat, \
         patch("ochat.ollama_embed", return_value=__import__("numpy").array([1.0], dtype="float32")):
        ochat.extract_facts(conn, "let's meet next Thursday", "sounds good", "default")
    system_message = mock_chat.call_args.args[0][0]["content"]
    assert "Current date/time:" in system_message


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


def test_handle_calendar_create_intent_confirm_text_shows_end_date_for_multi_day_event():
    cache = {"events": [], "fetched_at": None}
    intent_response = (
        '{"intent": "create", "title": "Conference", "start": "2026-06-22T09:00:00", '
        '"end": "2026-06-24T17:00:00", "notes": null}'
    )
    with patch("ochat.ollama_chat", return_value=intent_response), \
         patch("builtins.input", return_value="n") as mock_input, \
         patch("ochat.ochat_calendar.create_event") as mock_create:
        ochat.handle_calendar_create_intent("add a conference mon to wed", "Current date/time: ...", cache)
    mock_create.assert_not_called()
    confirm_text = mock_input.call_args.args[0]
    assert "Mon, Jun 22 2026" in confirm_text
    assert "Wed, Jun 24 2026" in confirm_text


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
