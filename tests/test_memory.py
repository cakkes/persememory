import sys
from pathlib import Path

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
