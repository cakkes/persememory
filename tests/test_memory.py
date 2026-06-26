import sys
from pathlib import Path

import pytest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import ochat


def test_cosine_similarity_identical_vectors_is_one():
    a = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    assert abs(ochat.cosine_similarity(a, a) - 1.0) < 1e-6


def test_cosine_similarity_orthogonal_vectors_is_zero():
    a = np.array([1.0, 0.0], dtype=np.float32)
    b = np.array([0.0, 1.0], dtype=np.float32)
    assert abs(ochat.cosine_similarity(a, b)) < 1e-6


def _fact(id_, text, vec):
    return {
        "id": id_,
        "text": text,
        "embedding": np.array(vec, dtype=np.float32),
        "source_thread": "default",
        "created_at": "2026-06-19T00:00:00+00:00",
    }


def test_top_k_facts_orders_by_similarity_and_respects_cutoff():
    query = np.array([1.0, 0.0], dtype=np.float32)
    facts = [
        _fact(1, "close match", [0.9, 0.1]),
        _fact(2, "exact match", [1.0, 0.0]),
        _fact(3, "unrelated", [0.0, 1.0]),  # similarity 0.0, below cutoff
    ]
    result = ochat.top_k_facts(query, facts, k=2, min_similarity=0.45)
    assert [f["id"] for f in result] == [2, 1]


def test_top_k_facts_returns_fewer_than_k_when_few_qualify():
    query = np.array([1.0, 0.0], dtype=np.float32)
    facts = [_fact(1, "only match", [1.0, 0.0])]
    result = ochat.top_k_facts(query, facts, k=8, min_similarity=0.45)
    assert len(result) == 1


def test_is_duplicate_fact_true_when_above_threshold():
    candidate = np.array([1.0, 0.0], dtype=np.float32)
    existing = [np.array([0.99, 0.01], dtype=np.float32)]
    assert ochat.is_duplicate_fact(candidate, existing, threshold=0.92) is True


def test_is_duplicate_fact_false_when_below_threshold():
    candidate = np.array([1.0, 0.0], dtype=np.float32)
    existing = [np.array([0.5, 0.5], dtype=np.float32)]
    assert ochat.is_duplicate_fact(candidate, existing, threshold=0.92) is False


def test_is_duplicate_fact_false_when_no_existing_facts():
    candidate = np.array([1.0, 0.0], dtype=np.float32)
    assert ochat.is_duplicate_fact(candidate, [], threshold=0.92) is False


def test_estimate_tokens_uses_four_chars_per_token():
    assert ochat.estimate_tokens("a" * 40) == 10


def test_truncate_messages_to_budget_keeps_most_recent_within_budget():
    messages = [
        {"role": "user", "content": "a" * 40},   # ~10 tokens
        {"role": "assistant", "content": "b" * 40},  # ~10 tokens
        {"role": "user", "content": "c" * 40},    # ~10 tokens
    ]
    result = ochat.truncate_messages_to_budget(messages, budget_tokens=25)
    assert [m["content"][0] for m in result] == ["b", "c"]


def test_truncate_messages_to_budget_always_keeps_newest_message():
    messages = [{"role": "user", "content": "x" * 1000}]
    result = ochat.truncate_messages_to_budget(messages, budget_tokens=1)
    assert len(result) == 1


def test_effective_history_budget_uses_max_budget_when_system_prompt_small():
    result = ochat.effective_history_budget("short system prompt", num_ctx=16384, response_reserve=2048, max_budget=8192)
    assert result == 8192


def test_effective_history_budget_shrinks_for_large_system_prompt():
    huge_system_prompt = "x" * (7000 * 4)  # ~7000 estimated tokens
    result = ochat.effective_history_budget(huge_system_prompt, num_ctx=16384, response_reserve=2048, max_budget=8192)
    assert result == 16384 - 7000 - 2048
    assert result < 8192


def test_effective_history_budget_never_negative():
    enormous_system_prompt = "x" * (50000 * 4)  # ~50000 estimated tokens, far over num_ctx
    result = ochat.effective_history_budget(enormous_system_prompt, num_ctx=16384, response_reserve=2048, max_budget=8192)
    assert result == 0


from datetime import datetime, timedelta, timezone
from unittest.mock import patch


def test_current_datetime_context_formats_local_time():
    fixed = datetime(2026, 6, 20, 18, 51, tzinfo=timezone(timedelta(hours=-7)))
    with patch("ochat.datetime") as mock_datetime:
        mock_datetime.now.return_value.astimezone.return_value = fixed
        result = ochat.current_datetime_context()
    assert result == "Current date/time: Saturday, June 20, 2026, 06:51 PM UTC-07:00"


import json


def test_load_thread_returns_fresh_thread_when_file_missing(tmp_path):
    path = tmp_path / "missing.json"
    thread = ochat.load_thread(path, "missing")
    assert thread == {"name": "missing", "messages": []}


def test_save_then_load_thread_round_trips(tmp_path):
    path = tmp_path / "work.json"
    thread = {"name": "work", "messages": [{"role": "user", "content": "hi", "ts": "t1"}]}
    ochat.save_thread(path, thread)
    loaded = ochat.load_thread(path, "work")
    assert loaded == thread


def test_load_thread_recovers_from_corrupt_file(tmp_path, capsys):
    path = tmp_path / "broken.json"
    path.write_text("{not valid json", encoding="utf-8")
    thread = ochat.load_thread(path, "broken")
    assert thread == {"name": "broken", "messages": []}
    corrupt_files = list(tmp_path.glob("broken.json.corrupt-*"))
    assert len(corrupt_files) == 1
    assert "corrupt" in capsys.readouterr().err


def test_insert_and_get_all_facts_round_trips(tmp_path):
    conn = ochat.init_db(tmp_path / "memory.db")
    embedding = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    ochat.insert_fact(conn, "likes terse answers", embedding, "default")
    facts = ochat.get_all_facts(conn)
    assert len(facts) == 1
    assert facts[0]["text"] == "likes terse answers"
    assert facts[0]["source_thread"] == "default"
    assert np.allclose(facts[0]["embedding"], embedding)


def test_delete_fact_removes_row_and_reports_success(tmp_path):
    conn = ochat.init_db(tmp_path / "memory.db")
    ochat.insert_fact(conn, "fact one", np.array([1.0], dtype=np.float32), "default")
    fact_id = ochat.get_all_facts(conn)[0]["id"]
    assert ochat.delete_fact(conn, fact_id) is True
    assert ochat.get_all_facts(conn) == []
    assert ochat.delete_fact(conn, fact_id) is False


from unittest.mock import MagicMock, patch


def test_think_param_off_is_false():
    assert ochat.think_param("off") is False


def test_think_param_passes_through_level_strings():
    assert ochat.think_param("medium") == "medium"


def test_check_ollama_ready_exits_when_unreachable():
    with patch("ochat.requests.get", side_effect=ochat.requests.RequestException("down")) as mock_get:
        with patch("ochat.sys.exit", side_effect=SystemExit) as mock_exit:
            try:
                ochat.check_ollama_ready()
            except SystemExit:
                pass
            mock_exit.assert_called_with(1)
            mock_get.assert_called_once()


def test_check_ollama_ready_exits_when_model_missing():
    version_response = MagicMock()
    version_response.raise_for_status.return_value = None
    tags_response = MagicMock()
    tags_response.json.return_value = {"models": [{"name": "gemma4:12b"}]}
    with patch("ochat.requests.get", side_effect=[version_response, tags_response]):
        with patch("ochat.sys.exit", side_effect=SystemExit) as mock_exit:
            try:
                ochat.check_ollama_ready()
            except SystemExit:
                pass
            mock_exit.assert_called_with(1)


def test_check_ollama_ready_succeeds_when_model_installed_with_default_latest_tag():
    version_response = MagicMock()
    version_response.raise_for_status.return_value = None
    tags_response = MagicMock()
    tags_response.json.return_value = {
        "models": [{"name": "gemma4:12b"}, {"name": "nomic-embed-text:latest"}]
    }
    with patch("ochat.requests.get", side_effect=[version_response, tags_response]):
        with patch("ochat.sys.exit") as mock_exit:
            ochat.check_ollama_ready()
            mock_exit.assert_not_called()


def test_ollama_embed_returns_numpy_array():
    fake_response = MagicMock()
    fake_response.json.return_value = {"embedding": [0.1, 0.2, 0.3]}
    fake_response.raise_for_status.return_value = None
    with patch("ochat.requests.post", return_value=fake_response) as mock_post:
        result = ochat.ollama_embed("hello")
    assert isinstance(result, np.ndarray)
    assert np.allclose(result, [0.1, 0.2, 0.3])
    mock_post.assert_called_once()
    assert mock_post.call_args.kwargs["json"]["model"] == ochat.EMBED_MODEL


def test_ollama_chat_streams_and_concatenates_content(capsys):
    fake_response = MagicMock()
    fake_response.raise_for_status.return_value = None
    fake_response.iter_lines.return_value = [
        json.dumps({"message": {"content": "Hel"}, "done": False}).encode(),
        json.dumps({"message": {"content": "lo"}, "done": False}).encode(),
        json.dumps({"message": {"content": ""}, "done": True}).encode(),
        json.dumps({"message": {"content": "EXTRA"}, "done": False}).encode(),
    ]
    with patch("ochat.requests.post", return_value=fake_response) as mock_post:
        result = ochat.ollama_chat([{"role": "user", "content": "hi"}], think=False, stream_to_stdout=True)
    assert result == "Hello"
    assert "EXTRA" not in result
    assert mock_post.call_args.kwargs["json"]["model"] == ochat.CHAT_MODEL
    assert mock_post.call_args.kwargs["json"]["think"] is False
    assert "Hello" in capsys.readouterr().out


def test_ollama_chat_sets_num_ctx_so_long_threads_dont_get_cut_off():
    fake_response = MagicMock()
    fake_response.raise_for_status.return_value = None
    fake_response.iter_lines.return_value = [
        json.dumps({"message": {"content": "hi"}, "done": True}).encode(),
    ]
    with patch("ochat.requests.post", return_value=fake_response) as mock_post:
        ochat.ollama_chat([{"role": "user", "content": "hi"}], think=False, stream_to_stdout=False)
    assert mock_post.call_args.kwargs["json"]["options"]["num_ctx"] == ochat.OLLAMA_NUM_CTX
    assert ochat.OLLAMA_NUM_CTX > ochat.CONTEXT_TOKEN_BUDGET


def test_ollama_chat_raises_response_truncated_error_when_done_reason_is_length():
    fake_response = MagicMock()
    fake_response.raise_for_status.return_value = None
    fake_response.iter_lines.return_value = [
        json.dumps({"message": {"content": "Hel"}, "done": False}).encode(),
        json.dumps({"message": {"content": "lo"}, "done": False}).encode(),
        json.dumps({"message": {"content": ""}, "done": True, "done_reason": "length"}).encode(),
    ]
    with patch("ochat.requests.post", return_value=fake_response):
        with pytest.raises(ochat.ResponseTruncatedError) as exc_info:
            ochat.ollama_chat([{"role": "user", "content": "hi"}], think=False, stream_to_stdout=False)
    assert exc_info.value.text == "Hello"


def test_extract_json_array_unchanged_behavior():
    text = '```json\n["fact one"]\n```'
    assert ochat._extract_json_array(text) == '["fact one"]'


def test_extract_facts_inserts_new_non_duplicate_facts(tmp_path):
    conn = ochat.init_db(tmp_path / "memory.db")
    with patch("ochat.EXTRACTION_LOG_PATH", tmp_path / "extraction.log"), \
         patch("ochat.ollama_chat", return_value='["likes terse answers"]'), \
         patch("ochat.ollama_embed", return_value=np.array([1.0, 0.0], dtype=np.float32)):
        ochat.extract_facts(conn, "be brief please", "ok, will do", "default")
    facts = ochat.get_all_facts(conn)
    assert len(facts) == 1
    assert facts[0]["text"] == "likes terse answers"


def test_extract_facts_skips_duplicate_facts(tmp_path):
    conn = ochat.init_db(tmp_path / "memory.db")
    ochat.insert_fact(conn, "likes terse answers", np.array([1.0, 0.0], dtype=np.float32), "default")
    with patch("ochat.EXTRACTION_LOG_PATH", tmp_path / "extraction.log"), \
         patch("ochat.ollama_chat", return_value='["likes terse answers"]'), \
         patch("ochat.ollama_embed", return_value=np.array([1.0, 0.0], dtype=np.float32)):
        ochat.extract_facts(conn, "be brief please", "ok, will do", "default")
    assert len(ochat.get_all_facts(conn)) == 1


def test_extract_facts_never_raises_and_logs_on_failure(tmp_path):
    conn = ochat.init_db(tmp_path / "memory.db")
    log_path = tmp_path / "extraction.log"
    with patch("ochat.EXTRACTION_LOG_PATH", log_path), \
         patch("ochat.ollama_chat", side_effect=RuntimeError("model unreachable")):
        ochat.extract_facts(conn, "hi", "hello", "default")  # must not raise
    assert log_path.exists()
    assert "model unreachable" in log_path.read_text(encoding="utf-8")


def test_extract_facts_handles_markdown_fenced_json(tmp_path):
    conn = ochat.init_db(tmp_path / "memory.db")
    with patch("ochat.EXTRACTION_LOG_PATH", tmp_path / "extraction.log"), \
         patch("ochat.ollama_chat", return_value='```json\n["likes terse answers"]\n```'), \
         patch("ochat.ollama_embed", return_value=np.array([1.0, 0.0], dtype=np.float32)):
        ochat.extract_facts(conn, "be brief please", "ok, will do", "default")
    facts = ochat.get_all_facts(conn)
    assert len(facts) == 1
    assert facts[0]["text"] == "likes terse answers"


def test_extract_facts_handles_prose_wrapped_json(tmp_path):
    conn = ochat.init_db(tmp_path / "memory.db")
    with patch("ochat.EXTRACTION_LOG_PATH", tmp_path / "extraction.log"), \
         patch("ochat.ollama_chat", return_value='Sure! Here are the facts: ["likes terse answers"]'), \
         patch("ochat.ollama_embed", return_value=np.array([1.0, 0.0], dtype=np.float32)):
        ochat.extract_facts(conn, "be brief please", "ok, will do", "default")
    facts = ochat.get_all_facts(conn)
    assert len(facts) == 1
    assert facts[0]["text"] == "likes terse answers"


def test_extract_facts_dedupes_within_same_call(tmp_path):
    """Verify that within-call dedup catches near-duplicate facts in the same model response."""
    conn = ochat.init_db(tmp_path / "memory.db")
    # Model returns two near-duplicate facts in one response
    with patch("ochat.EXTRACTION_LOG_PATH", tmp_path / "extraction.log"), \
         patch("ochat.ollama_chat", return_value='["likes terse answers", "prefers brief responses"]'), \
         patch("ochat.ollama_embed", side_effect=[
             np.array([1.0, 0.0], dtype=np.float32),    # first fact embedding
             np.array([0.99, 0.01], dtype=np.float32),  # second fact embedding (near-duplicate, > 0.92 similarity)
         ]):
        ochat.extract_facts(conn, "be brief please", "ok, will do", "default")
    # Only the first fact should be inserted; the second is a near-duplicate
    facts = ochat.get_all_facts(conn)
    assert len(facts) == 1
    assert facts[0]["text"] == "likes terse answers"


def test_extract_facts_includes_current_datetime_in_system_prompt(tmp_path):
    conn = ochat.init_db(tmp_path / "memory.db")
    with patch("ochat.EXTRACTION_LOG_PATH", tmp_path / "extraction.log"), \
         patch("ochat.ollama_chat", return_value="[]") as mock_chat, \
         patch("ochat.ollama_embed", return_value=np.array([1.0], dtype=np.float32)):
        ochat.extract_facts(conn, "let's meet next Thursday", "sounds good", "default")
    system_message = mock_chat.call_args.args[0][0]["content"]
    assert "Current date/time:" in system_message


def test_build_system_prompt_with_no_facts():
    prompt = ochat.build_system_prompt([])
    assert "Relevant memory" not in prompt


def test_build_system_prompt_includes_fact_bullets():
    facts = [
        {"id": 1, "text": "likes terse answers", "embedding": None, "source_thread": "x", "created_at": "t"},
    ]
    prompt = ochat.build_system_prompt(facts)
    assert "Relevant memory" in prompt
    assert "- likes terse answers" in prompt


def test_build_system_prompt_includes_tool_names_and_use_instruction_when_tools_given():
    tools = [
        {"type": "function", "function": {"name": "web_search", "description": "Search the web"}},
        {"type": "function", "function": {"name": "read_file", "description": "Read a file"}},
    ]
    prompt = ochat.build_system_prompt([], tools=tools)
    assert "web_search" in prompt
    assert "read_file" in prompt
    assert "tool" in prompt.lower()


def test_build_system_prompt_tool_section_does_not_use_echoed_phrase():
    # Gemma4 echoes "You have access to the following tools. Call them whenever..."
    # verbatim as its first response token stream. The tool section must use a
    # different format (e.g. XML tags) so the model treats it as config, not prose.
    tools = [{"type": "function", "function": {"name": "web_search", "description": "Search"}}]
    prompt = ochat.build_system_prompt([], tools=tools)
    assert "You have access to the following tools" not in prompt


def test_build_system_prompt_with_no_tools_has_no_tools_section():
    prompt = ochat.build_system_prompt([], tools=None)
    assert "web_search" not in prompt
    assert "Tools" not in prompt


def test_build_system_prompt_includes_current_datetime_context():
    prompt = ochat.build_system_prompt([])
    assert "Current date/time:" in prompt


def test_handle_turn_returns_none_and_does_not_save_on_request_failure(tmp_path):
    conn = ochat.init_db(tmp_path / "memory.db")
    thread = {"name": "t", "messages": []}
    path = tmp_path / "t.json"
    with patch("ochat.ollama_embed", side_effect=ochat.requests.RequestException("down")):
        result = ochat.handle_turn(conn, thread, path, "hello", "off")
    assert result is None
    assert thread["messages"] == []
    assert not path.exists()


def test_handle_turn_degrades_gracefully_when_retrieval_fails(tmp_path):
    conn = ochat.init_db(tmp_path / "memory.db")
    thread = {"name": "t", "messages": []}
    path = tmp_path / "t.json"
    with patch("ochat.ollama_embed", return_value=np.array([1.0], dtype=np.float32)), \
         patch("ochat.get_all_facts", side_effect=ochat.sqlite3.Error("disk I/O error")), \
         patch("ochat.ollama_chat", return_value="hi there"), \
         patch("ochat.extract_facts") as mock_extract:
        result = ochat.handle_turn(conn, thread, path, "hello", "off")
    assert result is not None
    result.join(timeout=2)
    assert len(thread["messages"]) == 2
    assert thread["messages"][0]["content"] == "hello"
    assert thread["messages"][1]["content"] == "hi there"
    assert path.exists()
    mock_extract.assert_called_once()


def test_handle_turn_saves_thread_and_starts_extraction_on_success(tmp_path):
    conn = ochat.init_db(tmp_path / "memory.db")
    thread = {"name": "t", "messages": []}
    path = tmp_path / "t.json"
    with patch("ochat.ollama_embed", return_value=np.array([1.0], dtype=np.float32)), \
         patch("ochat.ollama_chat", return_value="hi there"), \
         patch("ochat.extract_facts") as mock_extract:
        result = ochat.handle_turn(conn, thread, path, "hello", "off")
    assert result is not None
    result.join(timeout=2)
    assert len(thread["messages"]) == 2
    assert path.exists()
    mock_extract.assert_called_once()


def test_handle_turn_retries_once_with_smaller_window_after_truncated_reply(tmp_path, capsys):
    conn = ochat.init_db(tmp_path / "memory.db")
    # History large enough that budget//2 drops it, so retry has a genuinely smaller window.
    long_history = "y" * (ochat.CONTEXT_TOKEN_BUDGET // 2 * ochat.CHARS_PER_TOKEN + 4)
    thread = {
        "name": "t",
        "messages": [
            {"role": "user", "content": long_history, "ts": "t1"},
            {"role": "assistant", "content": "ok", "ts": "t2"},
        ],
    }
    path = tmp_path / "t.json"
    with patch("ochat.ollama_embed", return_value=np.array([1.0], dtype=np.float32)), \
         patch("ochat.ollama_chat", side_effect=[ochat.ResponseTruncatedError("partial"), "full reply"]) as mock_chat, \
         patch("ochat.extract_facts") as mock_extract:
        result = ochat.handle_turn(conn, thread, path, "hello", "off")
    assert result is not None
    result.join(timeout=2)
    assert mock_chat.call_count == 2
    assert thread["messages"][-1]["content"] == "full reply"
    assert path.exists()
    assert "cut off" in capsys.readouterr().err
    mock_extract.assert_called_once()


def test_handle_turn_saves_partial_reply_when_retry_also_truncated(tmp_path, capsys):
    conn = ochat.init_db(tmp_path / "memory.db")
    long_history = "y" * (ochat.CONTEXT_TOKEN_BUDGET // 2 * ochat.CHARS_PER_TOKEN + 4)
    thread = {
        "name": "t",
        "messages": [
            {"role": "user", "content": long_history, "ts": "t1"},
            {"role": "assistant", "content": "ok", "ts": "t2"},
        ],
    }
    path = tmp_path / "t.json"
    with patch("ochat.ollama_embed", return_value=np.array([1.0], dtype=np.float32)), \
         patch("ochat.ollama_chat", side_effect=[
             ochat.ResponseTruncatedError("first partial"),
             ochat.ResponseTruncatedError("second partial"),
         ]) as mock_chat, \
         patch("ochat.extract_facts") as mock_extract:
        result = ochat.handle_turn(conn, thread, path, "hello", "off")
    assert result is not None
    result.join(timeout=2)
    assert mock_chat.call_count == 2
    assert thread["messages"][-1]["content"] == "second partial"
    assert path.exists()
    assert capsys.readouterr().err.count("cut off") == 2
    mock_extract.assert_called_once()


# ── New tests covering the 10 review findings ─────────────────────────────────


def test_extract_json_array_handles_prose_with_brackets_before_fenced_json():
    # Bug 1: text.find('[') landed on prose bracket instead of JSON array bracket
    text = 'Here are [3] facts:\n```json\n["user prefers vim"]\n```'
    result = ochat._extract_json_array(text)
    assert json.loads(result) == ["user prefers vim"]


def test_extract_facts_recovers_partial_facts_from_truncated_extraction_response(tmp_path):
    # Bug 2: ResponseTruncatedError.text was discarded by the broad except
    conn = ochat.init_db(tmp_path / "memory.db")
    with patch("ochat.EXTRACTION_LOG_PATH", tmp_path / "extraction.log"), \
         patch("ochat.ollama_chat", side_effect=ochat.ResponseTruncatedError('["partial fact"]')), \
         patch("ochat.ollama_embed", return_value=np.array([1.0, 0.0], dtype=np.float32)):
        ochat.extract_facts(conn, "tell me something", "partial reply", "default")
    facts = ochat.get_all_facts(conn)
    assert len(facts) == 1
    assert facts[0]["text"] == "partial fact"


def test_handle_turn_embed_failure_error_message_says_embed_not_chat(tmp_path, capsys):
    # Bug 4: error said "chat request failed" when the embed call was the one that failed
    conn = ochat.init_db(tmp_path / "memory.db")
    thread = {"name": "t", "messages": []}
    path = tmp_path / "t.json"
    with patch("ochat.ollama_embed", side_effect=ochat.requests.RequestException("timeout")):
        ochat.handle_turn(conn, thread, path, "hello", "off")
    err = capsys.readouterr().err
    assert "embed" in err.lower()
    assert "chat" not in err.lower()


def test_handle_turn_skips_retry_when_halved_budget_yields_same_window(tmp_path, capsys):
    # Bug 5: with no history, budget//2 produces identical window — retry was a no-op
    conn = ochat.init_db(tmp_path / "memory.db")
    thread = {"name": "t", "messages": []}
    path = tmp_path / "t.json"
    with patch("ochat.ollama_embed", return_value=np.array([1.0], dtype=np.float32)), \
         patch("ochat.ollama_chat", side_effect=ochat.ResponseTruncatedError("partial reply")) as mock_chat, \
         patch("ochat.extract_facts"):
        result = ochat.handle_turn(conn, thread, path, "hello", "off")
    assert result is not None
    result.join(timeout=2)
    assert mock_chat.call_count == 1
    assert thread["messages"][-1]["content"] == "partial reply"
    assert "minimum" in capsys.readouterr().err


def test_insert_fact_does_not_auto_commit_to_db(tmp_path):
    # Opt 6: insert_fact must not commit so extract_facts can batch all inserts into one transaction
    conn = ochat.init_db(tmp_path / "memory.db")
    ochat.insert_fact(conn, "pending fact", np.array([1.0], dtype=np.float32), "default")
    conn.rollback()
    rows = conn.execute("SELECT id FROM facts").fetchall()
    assert len(rows) == 0


def test_get_all_facts_returns_inserted_fact_from_cache_before_commit(tmp_path):
    # Opt 6: cache must expose the fact immediately after insert even before a DB commit
    conn = ochat.init_db(tmp_path / "memory.db")
    ochat.insert_fact(conn, "uncommitted fact", np.array([1.0], dtype=np.float32), "default")
    facts = ochat.get_all_facts(conn)
    assert any(f["text"] == "uncommitted fact" for f in facts)


def test_save_thread_writes_compact_json_without_indentation(tmp_path):
    # Opt 9: indent=2 bloats files by ~40%; machine-read-only files need no pretty-printing
    path = tmp_path / "test.json"
    thread = {"name": "test", "messages": [{"role": "user", "content": "hi"}]}
    ochat.save_thread(path, thread)
    raw = path.read_text()
    assert "  " not in raw


def test_build_system_prompt_warns_that_older_messages_may_reference_a_different_date():
    # Bug: threads spanning multiple days contain old assistant messages that state a prior
    # date. Gemma4's conversation-coherence bias causes it to repeat the old date rather
    # than the correct one from the system prompt. The system prompt must explicitly instruct
    # the model to disregard date/time references in history and use the injected value.
    prompt = ochat.build_system_prompt([])
    assert "older messages" in prompt.lower() or "previous messages" in prompt.lower() or "earlier messages" in prompt.lower()


def test_handle_turn_prefixes_user_message_with_datetime_in_payload_but_not_in_thread(tmp_path):
    # Bug: system prompt date instruction alone is insufficient — Gemma4's conversation-
    # coherence bias overrides it when old messages in history state a different date.
    # Injecting the date directly into the outgoing user message (not persisted) gives the
    # model an authoritative date signal right before it generates its reply.
    conn = ochat.init_db(tmp_path / "memory.db")
    thread = {"name": "t", "messages": [
        {"role": "user", "content": "hi", "ts": "t1"},
        {"role": "assistant", "content": "It is 9:33 AM on June 21, 2026.", "ts": "t2"},
    ]}
    path = tmp_path / "t.json"
    captured_payloads = []

    def fake_chat(messages, **kwargs):
        captured_payloads.append(messages)
        return "today reply"

    with patch("ochat.ollama_embed", return_value=np.array([1.0], dtype=np.float32)), \
         patch("ochat.ollama_chat", side_effect=fake_chat), \
         patch("ochat.extract_facts"):
        result = ochat.handle_turn(conn, thread, path, "what day is it?", "off")

    result.join(timeout=2)
    # The payload's last user message must contain the injected datetime
    last_user_msg = next(
        m for m in reversed(captured_payloads[0]) if m["role"] == "user"
    )
    assert "current date" in last_user_msg["content"].lower() or "date/time" in last_user_msg["content"].lower()
    # The thread must store the original clean user input, not the dated version
    assert thread["messages"][-2]["content"] == "what day is it?"


def test_handle_turn_prints_reply_to_stdout_when_tools_active_and_model_makes_no_tool_calls(tmp_path, capsys):
    # Bug: when tools are registered, handle_turn uses ollama_chat_raw (non-streaming)
    # for tool detection. If the model returns no tool_calls, the reply text came from
    # a non-streaming call and was never printed — the user would see "ochat> " then silence.
    import ochat_tools
    ochat_tools.clear_registry()
    ochat_tools.register(ochat_tools.ToolDef(
        name="dummy_tool", description="", parameters={},
        fn=lambda: "result",
        dangerous=False,
    ))
    conn = ochat.init_db(tmp_path / "memory.db")
    thread = {"name": "t", "messages": []}
    path = tmp_path / "t.json"
    with patch("ochat.ollama_embed", return_value=np.array([1.0], dtype=np.float32)), \
         patch("ochat.ollama_chat_raw", return_value=("hello from model", None)), \
         patch("ochat.extract_facts"):
        result = ochat.handle_turn(conn, thread, path, "hi", "off")
    result.join(timeout=2)
    ochat_tools.clear_registry()
    out = capsys.readouterr().out
    assert "hello from model" in out
