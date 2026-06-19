# ochat Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build `ochat`, a single-file Python CLI that replaces `ollama run gemma4:12b` with a tool that resumes named conversation threads across terminal sessions and recalls long-term facts via semantic search.

**Architecture:** One executable script, `ochat.py`, run via `uv run --script` (PEP 723 inline deps — no venv setup). Conversation threads persist as JSON files; long-term facts persist in a single SQLite database with embeddings for cosine-similarity retrieval. A background daemon thread extracts durable facts after each reply without blocking the interactive loop.

**Tech Stack:** Python 3.11+, `uv` (script runner + ephemeral deps), `numpy` (vector math), `requests` (Ollama HTTP API), `sqlite3` (stdlib), `pytest` (tests, run via `uv run --with pytest ...`).

## Global Constraints

- Single file `ochat.py` at repo root — no separate package/module split (explicit spec decision).
- Dependencies limited to `numpy` and `requests`, declared inline via PEP 723; no `requirements.txt`, no `pyproject.toml`, no venv.
- No config file — all tunables are module-level constants in `ochat.py`.
- No background service/daemon process — the tool only runs while invoked; the only background work is a single daemon `threading.Thread` per turn, joined (with timeout) before the process exits.
- Default `think` is `off` for the main chat call; the model is `gemma4:12b`; the embedding model is `nomic-embed-text`.
- Data directory: `~/.local/share/ochat/` containing `threads/<name>.json`, `memory.db`, `extraction.log`.
- Retrieval: top 8 facts, similarity >= 0.45. Dedup on insert: skip if similarity > 0.92 to an existing fact.
- Context budget: 8192 tokens, estimated as `len(text) // 4` (no real tokenizer).
- Ollama base URL: `http://127.0.0.1:11434`.
- Never auto-pull models or auto-overwrite a corrupt thread file; always tell the user and exit/recover visibly.

---

### Task 1: Project scaffold + cosine similarity + top-k retrieval

**Files:**
- Create: `ochat.py`
- Test: `tests/test_memory.py`

**Interfaces:**
- Produces: `cosine_similarity(a: np.ndarray, b: np.ndarray) -> float`
- Produces: `top_k_facts(query_embedding: np.ndarray, facts: list[dict], k: int = RETRIEVAL_TOP_K, min_similarity: float = RETRIEVAL_MIN_SIMILARITY) -> list[dict]` where each fact dict has keys `id, text, embedding, source_thread, created_at`.

- [ ] **Step 1: Create the scaffold file**

Create `ochat.py`:

```python
#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy", "requests"]
# ///
"""ochat: persistent-memory terminal chat for Ollama."""

import argparse
import json
import os
import sqlite3
import sys
import threading
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import requests

OLLAMA_URL = "http://127.0.0.1:11434"
CHAT_MODEL = "gemma4:12b"
EMBED_MODEL = "nomic-embed-text"

DATA_DIR = Path.home() / ".local" / "share" / "ochat"
THREADS_DIR = DATA_DIR / "threads"
MEMORY_DB_PATH = DATA_DIR / "memory.db"
EXTRACTION_LOG_PATH = DATA_DIR / "extraction.log"

CONTEXT_TOKEN_BUDGET = 8192
CHARS_PER_TOKEN = 4
RETRIEVAL_TOP_K = 8
RETRIEVAL_MIN_SIMILARITY = 0.45
DEDUP_SIMILARITY_THRESHOLD = 0.92
DEFAULT_THINK = "off"
EXTRACTION_JOIN_TIMEOUT_SECONDS = 5
```

- [ ] **Step 2: Make it executable**

Run: `chmod +x /Users/developer/ochat/ochat.py`

- [ ] **Step 3: Create the test directory and write the failing test**

Create `tests/test_memory.py`:

```python
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
```

- [ ] **Step 4: Run the test to verify it fails**

Run: `uv run --with pytest --with numpy --with requests pytest tests/test_memory.py -v`
Expected: FAIL with `AttributeError: module 'ochat' has no attribute 'cosine_similarity'`

- [ ] **Step 5: Implement `cosine_similarity`**

Append to `ochat.py`:

```python
def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0.0:
        return 0.0
    return float(np.dot(a, b) / denom)
```

- [ ] **Step 6: Run the test to verify it passes**

Run: `uv run --with pytest --with numpy --with requests pytest tests/test_memory.py -v`
Expected: PASS (2 tests)

- [ ] **Step 7: Write the failing test for top-k retrieval**

Append to `tests/test_memory.py`:

```python
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
```

- [ ] **Step 8: Run the test to verify it fails**

Run: `uv run --with pytest --with numpy --with requests pytest tests/test_memory.py -v`
Expected: FAIL with `AttributeError: module 'ochat' has no attribute 'top_k_facts'`

- [ ] **Step 9: Implement `top_k_facts`**

Append to `ochat.py`:

```python
def top_k_facts(query_embedding, facts, k=RETRIEVAL_TOP_K, min_similarity=RETRIEVAL_MIN_SIMILARITY):
    scored = []
    for fact in facts:
        similarity = cosine_similarity(query_embedding, fact["embedding"])
        if similarity >= min_similarity:
            scored.append((similarity, fact))
    scored.sort(key=lambda pair: pair[0], reverse=True)
    return [fact for _, fact in scored[:k]]
```

- [ ] **Step 10: Run the test to verify it passes**

Run: `uv run --with pytest --with numpy --with requests pytest tests/test_memory.py -v`
Expected: PASS (4 tests)

- [ ] **Step 11: Commit**

```bash
cd /Users/developer/ochat
git add ochat.py tests/test_memory.py
git commit -m "feat: add cosine similarity and top-k fact retrieval"
```

---

### Task 2: Fact dedup logic

**Files:**
- Modify: `ochat.py`
- Test: `tests/test_memory.py`

**Interfaces:**
- Consumes: `cosine_similarity` (Task 1)
- Produces: `is_duplicate_fact(candidate_embedding: np.ndarray, existing_embeddings: list[np.ndarray], threshold: float = DEDUP_SIMILARITY_THRESHOLD) -> bool`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_memory.py`:

```python
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
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `uv run --with pytest --with numpy --with requests pytest tests/test_memory.py -v`
Expected: FAIL with `AttributeError: module 'ochat' has no attribute 'is_duplicate_fact'`

- [ ] **Step 3: Implement `is_duplicate_fact`**

Append to `ochat.py`:

```python
def is_duplicate_fact(candidate_embedding, existing_embeddings, threshold=DEDUP_SIMILARITY_THRESHOLD):
    for embedding in existing_embeddings:
        if cosine_similarity(candidate_embedding, embedding) > threshold:
            return True
    return False
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `uv run --with pytest --with numpy --with requests pytest tests/test_memory.py -v`
Expected: PASS (7 tests)

- [ ] **Step 5: Commit**

```bash
cd /Users/developer/ochat
git add ochat.py tests/test_memory.py
git commit -m "feat: add fact dedup check"
```

---

### Task 3: Token-budget sliding window truncation

**Files:**
- Modify: `ochat.py`
- Test: `tests/test_memory.py`

**Interfaces:**
- Produces: `estimate_tokens(text: str) -> int`
- Produces: `truncate_messages_to_budget(messages: list[dict], budget_tokens: int = CONTEXT_TOKEN_BUDGET) -> list[dict]` where each message dict has at least a `content` key.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_memory.py`:

```python
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
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `uv run --with pytest --with numpy --with requests pytest tests/test_memory.py -v`
Expected: FAIL with `AttributeError: module 'ochat' has no attribute 'estimate_tokens'`

- [ ] **Step 3: Implement both functions**

Append to `ochat.py`:

```python
def estimate_tokens(text: str) -> int:
    return max(1, len(text) // CHARS_PER_TOKEN)


def truncate_messages_to_budget(messages, budget_tokens=CONTEXT_TOKEN_BUDGET):
    selected = []
    used = 0
    for message in reversed(messages):
        cost = estimate_tokens(message["content"])
        if selected and used + cost > budget_tokens:
            break
        selected.append(message)
        used += cost
    selected.reverse()
    return selected
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `uv run --with pytest --with numpy --with requests pytest tests/test_memory.py -v`
Expected: PASS (10 tests)

- [ ] **Step 5: Commit**

```bash
cd /Users/developer/ochat
git add ochat.py tests/test_memory.py
git commit -m "feat: add token-budget sliding window truncation"
```

---

### Task 4: Thread JSON load/save with corrupt-file recovery

**Files:**
- Modify: `ochat.py`
- Test: `tests/test_memory.py`

**Interfaces:**
- Produces: `thread_path(name: str) -> Path`
- Produces: `load_thread(path: Path, name: str) -> dict` returning `{"name": str, "messages": list[dict]}`
- Produces: `save_thread(path: Path, thread: dict) -> None`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_memory.py`:

```python
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
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `uv run --with pytest --with numpy --with requests pytest tests/test_memory.py -v`
Expected: FAIL with `AttributeError: module 'ochat' has no attribute 'load_thread'`

- [ ] **Step 3: Implement thread storage functions**

Append to `ochat.py`:

```python
def thread_path(name: str) -> Path:
    return THREADS_DIR / f"{name}.json"


def load_thread(path: Path, name: str) -> dict:
    if not path.exists():
        return {"name": name, "messages": []}
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict) or "messages" not in data:
            raise ValueError("missing 'messages' key")
        return data
    except (json.JSONDecodeError, ValueError) as exc:
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        corrupt_path = path.with_name(f"{path.name}.corrupt-{timestamp}")
        path.rename(corrupt_path)
        print(
            f"warning: {path.name} was corrupt ({exc}); moved to "
            f"{corrupt_path.name} and starting a fresh thread",
            file=sys.stderr,
        )
        return {"name": name, "messages": []}


def save_thread(path: Path, thread: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(".tmp")
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(thread, f, indent=2)
    os.replace(tmp_path, path)
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `uv run --with pytest --with numpy --with requests pytest tests/test_memory.py -v`
Expected: PASS (13 tests)

- [ ] **Step 5: Commit**

```bash
cd /Users/developer/ochat
git add ochat.py tests/test_memory.py
git commit -m "feat: add thread JSON load/save with corrupt-file recovery"
```

---

### Task 5: SQLite long-term memory store

**Files:**
- Modify: `ochat.py`
- Test: `tests/test_memory.py`

**Interfaces:**
- Consumes: `MEMORY_DB_PATH` constant (Task 1)
- Produces: `init_db(db_path: Path) -> sqlite3.Connection`
- Produces: `insert_fact(conn, text: str, embedding: np.ndarray, source_thread: str) -> None`
- Produces: `get_all_facts(conn) -> list[dict]` (same shape consumed by `top_k_facts`)
- Produces: `delete_fact(conn, fact_id: int) -> bool`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_memory.py`:

```python
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
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `uv run --with pytest --with numpy --with requests pytest tests/test_memory.py -v`
Expected: FAIL with `AttributeError: module 'ochat' has no attribute 'init_db'`

- [ ] **Step 3: Implement the SQLite store**

Append to `ochat.py`:

```python
def init_db(db_path: Path) -> sqlite3.Connection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path, check_same_thread=False)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS facts (
            id INTEGER PRIMARY KEY,
            text TEXT NOT NULL,
            embedding BLOB NOT NULL,
            source_thread TEXT,
            created_at TEXT NOT NULL
        )
        """
    )
    conn.commit()
    return conn


def insert_fact(conn, text, embedding, source_thread):
    conn.execute(
        "INSERT INTO facts (text, embedding, source_thread, created_at) VALUES (?, ?, ?, ?)",
        (
            text,
            np.asarray(embedding, dtype=np.float32).tobytes(),
            source_thread,
            datetime.now(timezone.utc).isoformat(),
        ),
    )
    conn.commit()


def get_all_facts(conn):
    rows = conn.execute(
        "SELECT id, text, embedding, source_thread, created_at FROM facts"
    ).fetchall()
    return [
        {
            "id": row[0],
            "text": row[1],
            "embedding": np.frombuffer(row[2], dtype=np.float32),
            "source_thread": row[3],
            "created_at": row[4],
        }
        for row in rows
    ]


def delete_fact(conn, fact_id):
    cursor = conn.execute("DELETE FROM facts WHERE id = ?", (fact_id,))
    conn.commit()
    return cursor.rowcount > 0
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `uv run --with pytest --with numpy --with requests pytest tests/test_memory.py -v`
Expected: PASS (15 tests)

- [ ] **Step 5: Commit**

```bash
cd /Users/developer/ochat
git add ochat.py tests/test_memory.py
git commit -m "feat: add SQLite long-term fact store"
```

---

### Task 6: Ollama HTTP client (readiness check, embeddings, chat)

**Files:**
- Modify: `ochat.py`
- Test: `tests/test_memory.py`

**Interfaces:**
- Consumes: `OLLAMA_URL`, `CHAT_MODEL`, `EMBED_MODEL` constants (Task 1)
- Produces: `check_ollama_ready() -> None` (calls `sys.exit(1)` on failure)
- Produces: `ollama_embed(text: str) -> np.ndarray`
- Produces: `think_param(level: str) -> bool | str`
- Produces: `ollama_chat(messages: list[dict], think: bool | str = False, stream_to_stdout: bool = True) -> str`

- [ ] **Step 1: Write the failing test for `think_param`**

Append to `tests/test_memory.py`:

```python
from unittest.mock import MagicMock, patch


def test_think_param_off_is_false():
    assert ochat.think_param("off") is False


def test_think_param_passes_through_level_strings():
    assert ochat.think_param("medium") == "medium"
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `uv run --with pytest --with numpy --with requests pytest tests/test_memory.py -v`
Expected: FAIL with `AttributeError: module 'ochat' has no attribute 'think_param'`

- [ ] **Step 3: Implement `think_param` and `check_ollama_ready`**

Append to `ochat.py`:

```python
def think_param(level: str):
    if level == "off":
        return False
    if level in ("on", "true"):
        return True
    return level


def check_ollama_ready():
    try:
        requests.get(f"{OLLAMA_URL}/api/version", timeout=3).raise_for_status()
    except requests.RequestException:
        print(
            f"error: Ollama isn't reachable at {OLLAMA_URL} — start it with `ollama serve`",
            file=sys.stderr,
        )
        sys.exit(1)
    try:
        tags = requests.get(f"{OLLAMA_URL}/api/tags", timeout=3).json()
    except requests.RequestException:
        print("error: failed to query installed Ollama models", file=sys.stderr)
        sys.exit(1)
    installed = {model["name"] for model in tags.get("models", [])}
    missing = [model for model in (CHAT_MODEL, EMBED_MODEL) if model not in installed]
    for model in missing:
        print(
            f"error: required model '{model}' is not installed — run `ollama pull {model}`",
            file=sys.stderr,
        )
    if missing:
        sys.exit(1)
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `uv run --with pytest --with numpy --with requests pytest tests/test_memory.py -v`
Expected: PASS (17 tests)

- [ ] **Step 5: Write the failing test for `check_ollama_ready` failure paths**

Append to `tests/test_memory.py`:

```python
def test_check_ollama_ready_exits_when_unreachable():
    with patch("ochat.requests.get", side_effect=ochat.requests.RequestException("down")):
        with patch("ochat.sys.exit", side_effect=SystemExit) as mock_exit:
            try:
                ochat.check_ollama_ready()
            except SystemExit:
                pass
            mock_exit.assert_called_with(1)


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
```

- [ ] **Step 6: Run the test to verify it fails**

Run: `uv run --with pytest --with numpy --with requests pytest tests/test_memory.py -v`
Expected: FAIL — `check_ollama_ready` currently calls the real `sys.exit`, so without the patch in place this would raise `requests.RequestException` instead of being caught (confirms the test exercises real behavior). With the implementation from Step 3 already in place, this actually PASSES immediately since the try/except + `sys.exit` call already exist. Treat this as a regression-protection test rather than a red/green cycle, and proceed to Step 7.

- [ ] **Step 7: Run the test to verify it passes**

Run: `uv run --with pytest --with numpy --with requests pytest tests/test_memory.py -v`
Expected: PASS (19 tests)

- [ ] **Step 8: Write the failing test for `ollama_embed`**

Append to `tests/test_memory.py`:

```python
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
```

- [ ] **Step 9: Run the test to verify it fails**

Run: `uv run --with pytest --with numpy --with requests pytest tests/test_memory.py -v`
Expected: FAIL with `AttributeError: module 'ochat' has no attribute 'ollama_embed'`

- [ ] **Step 10: Implement `ollama_embed` and `ollama_chat`**

Append to `ochat.py`:

```python
def ollama_embed(text: str) -> np.ndarray:
    response = requests.post(
        f"{OLLAMA_URL}/api/embeddings",
        json={"model": EMBED_MODEL, "prompt": text},
        timeout=30,
    )
    response.raise_for_status()
    return np.array(response.json()["embedding"], dtype=np.float32)


def ollama_chat(messages, think=False, stream_to_stdout=True):
    response = requests.post(
        f"{OLLAMA_URL}/api/chat",
        json={"model": CHAT_MODEL, "messages": messages, "stream": True, "think": think},
        stream=True,
        timeout=120,
    )
    response.raise_for_status()
    pieces = []
    for line in response.iter_lines():
        if not line:
            continue
        chunk = json.loads(line)
        piece = chunk.get("message", {}).get("content", "")
        if piece:
            pieces.append(piece)
            if stream_to_stdout:
                print(piece, end="", flush=True)
        if chunk.get("done"):
            break
    if stream_to_stdout:
        print()
    return "".join(pieces)
```

- [ ] **Step 11: Run the test to verify it passes**

Run: `uv run --with pytest --with numpy --with requests pytest tests/test_memory.py -v`
Expected: PASS (20 tests)

- [ ] **Step 12: Commit**

```bash
cd /Users/developer/ochat
git add ochat.py tests/test_memory.py
git commit -m "feat: add Ollama HTTP client (readiness check, embeddings, chat)"
```

---

### Task 7: Background fact extraction

**Files:**
- Modify: `ochat.py`
- Test: `tests/test_memory.py`

**Interfaces:**
- Consumes: `ollama_chat`, `ollama_embed` (Task 6), `is_duplicate_fact` (Task 2), `insert_fact`, `get_all_facts` (Task 5), `EXTRACTION_LOG_PATH` (Task 1)
- Produces: `log_extraction_error(exc: Exception) -> None`
- Produces: `extract_facts(conn, user_message: str, assistant_message: str, source_thread: str) -> None` (never raises)

- [ ] **Step 1: Write the failing test**

Append to `tests/test_memory.py`:

```python
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
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `uv run --with pytest --with numpy --with requests pytest tests/test_memory.py -v`
Expected: FAIL with `AttributeError: module 'ochat' has no attribute 'extract_facts'`

- [ ] **Step 3: Implement extraction**

Append to `ochat.py`:

```python
EXTRACTION_PROMPT = (
    "Given this exchange, list any new durable facts or preferences about the "
    "user worth remembering long-term. Respond with ONLY a JSON array of short "
    'fact strings, e.g. ["prefers terse answers"]. If nothing is worth '
    "remembering, respond with []."
)


def log_extraction_error(exc: Exception) -> None:
    EXTRACTION_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(EXTRACTION_LOG_PATH, "a", encoding="utf-8") as f:
        f.write(f"{datetime.now(timezone.utc).isoformat()} {exc!r}\n")


def extract_facts(conn, user_message: str, assistant_message: str, source_thread: str) -> None:
    try:
        exchange = f"User: {user_message}\nAssistant: {assistant_message}"
        reply = ollama_chat(
            [
                {"role": "system", "content": EXTRACTION_PROMPT},
                {"role": "user", "content": exchange},
            ],
            think=False,
            stream_to_stdout=False,
        )
        candidate_facts = json.loads(reply)
        if not isinstance(candidate_facts, list):
            return
        existing_embeddings = [fact["embedding"] for fact in get_all_facts(conn)]
        for fact_text in candidate_facts:
            if not isinstance(fact_text, str) or not fact_text.strip():
                continue
            embedding = ollama_embed(fact_text)
            if is_duplicate_fact(embedding, existing_embeddings):
                continue
            insert_fact(conn, fact_text, embedding, source_thread)
            existing_embeddings.append(embedding)
    except Exception as exc:  # extraction must never crash the main loop
        log_extraction_error(exc)
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `uv run --with pytest --with numpy --with requests pytest tests/test_memory.py -v`
Expected: PASS (23 tests)

- [ ] **Step 5: Commit**

```bash
cd /Users/developer/ochat
git add ochat.py tests/test_memory.py
git commit -m "feat: add background fact extraction with dedup and error logging"
```

---

### Task 8: System prompt builder + main chat loop

**Files:**
- Modify: `ochat.py`
- Test: `tests/test_memory.py`

**Interfaces:**
- Consumes: `check_ollama_ready`, `ollama_embed`, `ollama_chat`, `think_param` (Task 6), `top_k_facts` (Task 1), `init_db`, `get_all_facts` (Task 5), `load_thread`, `save_thread`, `thread_path` (Task 4), `truncate_messages_to_budget` (Task 3), `extract_facts` (Task 7)
- Produces: `build_system_prompt(relevant_facts: list[dict]) -> str`
- Produces: `handle_turn(conn, thread: dict, path: Path, user_input: str, think: str) -> threading.Thread | None` — processes one turn; returns the background extraction thread on success, or `None` if the request failed (in which case `thread` is left unmodified and nothing is written to disk)
- Produces: `run_chat_loop(thread_name: str, think: str) -> None`

Note: the spec requires a failed chat call to print an error and leave thread history untouched so the user can retry. A bare call to `ollama_chat`/`ollama_embed` directly inside the `while True` loop would let a `requests.RequestException` propagate out and crash the whole REPL instead. `handle_turn` exists specifically to contain that failure per-turn — `run_chat_loop` only handles reading input and bookkeeping the extraction thread across turns.

- [ ] **Step 1: Write the failing tests for `build_system_prompt` and `handle_turn`**

Append to `tests/test_memory.py`:

```python
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


def test_handle_turn_returns_none_and_does_not_save_on_request_failure(tmp_path):
    conn = ochat.init_db(tmp_path / "memory.db")
    thread = {"name": "t", "messages": []}
    path = tmp_path / "t.json"
    with patch("ochat.ollama_embed", side_effect=ochat.requests.RequestException("down")):
        result = ochat.handle_turn(conn, thread, path, "hello", "off")
    assert result is None
    assert thread["messages"] == []
    assert not path.exists()


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
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `uv run --with pytest --with numpy --with requests pytest tests/test_memory.py -v`
Expected: FAIL with `AttributeError: module 'ochat' has no attribute 'build_system_prompt'`

- [ ] **Step 3: Implement `build_system_prompt`, `handle_turn`, and `run_chat_loop`**

Append to `ochat.py`:

```python
def build_system_prompt(relevant_facts):
    base = "You are a helpful assistant talking with the user in their terminal."
    if not relevant_facts:
        return base
    bullets = "\n".join(f"- {fact['text']}" for fact in relevant_facts)
    return f"{base}\n\nRelevant memory:\n{bullets}"


def handle_turn(conn, thread, path, user_input, think):
    try:
        query_embedding = ollama_embed(user_input)
        relevant = top_k_facts(query_embedding, get_all_facts(conn))
        system_prompt = build_system_prompt(relevant)
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


def run_chat_loop(thread_name: str, think: str) -> None:
    check_ollama_ready()
    conn = init_db(MEMORY_DB_PATH)
    path = thread_path(thread_name)
    thread = load_thread(path, thread_name)
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
            pending_extraction = handle_turn(conn, thread, path, user_input, think)
    finally:
        if pending_extraction is not None:
            pending_extraction.join(timeout=EXTRACTION_JOIN_TIMEOUT_SECONDS)
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `uv run --with pytest --with numpy --with requests pytest tests/test_memory.py -v`
Expected: PASS (27 tests)

- [ ] **Step 5: Manual check that the loop starts and exits cleanly**

Run: `printf '' | uv run --script ochat.py` (empty stdin triggers immediate EOF)
Expected: prints `you> ` then exits with code 0, no traceback. (This will fail with a connection error if Ollama/models aren't ready yet — that's expected until Task 10's setup step; confirm no *Python* traceback/crash beyond the handled readiness error.)

- [ ] **Step 6: Commit**

```bash
cd /Users/developer/ochat
git add ochat.py tests/test_memory.py
git commit -m "feat: add system prompt builder and main interactive chat loop"
```

---

### Task 9: CLI subcommands (`threads`, `memory list`, `memory forget`) and entrypoint

**Files:**
- Modify: `ochat.py`

**Interfaces:**
- Consumes: `THREADS_DIR` (Task 1), `load_thread` (Task 4), `init_db`, `get_all_facts`, `delete_fact` (Task 5), `run_chat_loop` (Task 8), `DEFAULT_THINK` (Task 1)
- Produces: `cmd_threads() -> None`, `cmd_memory_list() -> None`, `cmd_memory_forget(fact_id: int) -> None`, `main() -> None`

- [ ] **Step 1: Implement the subcommands and argument parser**

Append to `ochat.py`:

```python
def cmd_threads() -> None:
    if not THREADS_DIR.exists() or not any(THREADS_DIR.glob("*.json")):
        print("no threads yet")
        return
    for path in sorted(THREADS_DIR.glob("*.json")):
        thread = load_thread(path, path.stem)
        modified = datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc).isoformat()
        print(f"{path.stem}\t{len(thread['messages'])} messages\tlast updated {modified}")


def cmd_memory_list() -> None:
    conn = init_db(MEMORY_DB_PATH)
    facts = get_all_facts(conn)
    if not facts:
        print("no facts stored yet")
        return
    for fact in facts:
        print(f"[{fact['id']}] {fact['text']}  (from '{fact['source_thread']}', {fact['created_at']})")


def cmd_memory_forget(fact_id: int) -> None:
    conn = init_db(MEMORY_DB_PATH)
    if delete_fact(conn, fact_id):
        print(f"deleted fact {fact_id}")
    else:
        print(f"no fact with id {fact_id}", file=sys.stderr)
        sys.exit(1)


def main() -> None:
    parser = argparse.ArgumentParser(prog="ochat")
    parser.add_argument("--thread", default="default")
    parser.add_argument("--think", default=DEFAULT_THINK)
    subparsers = parser.add_subparsers(dest="command")
    subparsers.add_parser("threads")
    memory_parser = subparsers.add_parser("memory")
    memory_sub = memory_parser.add_subparsers(dest="memory_command")
    memory_sub.add_parser("list")
    forget_parser = memory_sub.add_parser("forget")
    forget_parser.add_argument("fact_id", type=int)

    args = parser.parse_args()

    if args.command == "threads":
        cmd_threads()
    elif args.command == "memory":
        if args.memory_command == "list":
            cmd_memory_list()
        elif args.memory_command == "forget":
            cmd_memory_forget(args.fact_id)
        else:
            memory_parser.print_help()
    else:
        run_chat_loop(args.thread, args.think)


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Manual check — `threads` with no data**

Run: `HOME=/tmp/ochat-manual-check uv run --script ochat.py threads`
Expected: prints `no threads yet`

- [ ] **Step 3: Manual check — `memory list` with no data**

Run: `HOME=/tmp/ochat-manual-check uv run --script ochat.py memory list`
Expected: prints `no facts stored yet`

- [ ] **Step 4: Manual check — `memory forget` on a nonexistent id**

Run: `HOME=/tmp/ochat-manual-check uv run --script ochat.py memory forget 999`
Expected: prints `no fact with id 999` to stderr, exit code 1

- [ ] **Step 5: Clean up the manual-check scratch directory**

Run: `rm -rf /tmp/ochat-manual-check`

- [ ] **Step 6: Run the full test suite once more**

Run: `uv run --with pytest --with numpy --with requests pytest tests/test_memory.py -v`
Expected: PASS (27 tests)

- [ ] **Step 7: Commit**

```bash
cd /Users/developer/ochat
git add ochat.py
git commit -m "feat: add threads/memory CLI subcommands and entrypoint"
```

---

### Task 10: Install, pull embedding model, end-to-end smoke test

**Files:**
- None created/modified (operational steps only)

**Interfaces:**
- Consumes: the complete `ochat.py` from Tasks 1-9

- [ ] **Step 1: Pull the embedding model**

Run: `ollama pull nomic-embed-text`
Expected: download completes, `ollama list` now shows both `gemma4:12b` and `nomic-embed-text`

- [ ] **Step 2: Symlink onto PATH**

Run:
```bash
mkdir -p ~/.local/bin
ln -sf /Users/developer/ochat/ochat.py ~/.local/bin/ochat
```
Expected: `which ochat` resolves to `/Users/developer/.local/bin/ochat`

- [ ] **Step 3: Smoke test 1 — new thread, exchange messages, quit, resume**

Run interactively: `ochat --thread smoketest`, type a message (e.g. "My favorite color is teal."), wait for a reply, then send EOF (Ctrl-D) to quit. Then run `ochat --thread smoketest` again and ask "What's my favorite color?"
Expected: the second run's reply references teal, sourced from the resumed thread history (not yet from long-term memory, since this is still within the same short-term window).

- [ ] **Step 4: Smoke test 2 — confirm fact extraction**

Run: `ochat memory list`
Expected: at least one stored fact referencing the favorite color, attributed to `source_thread = smoketest`. (Allow a few seconds after the Step 3 conversation for the background extraction thread to complete; if nothing appears, check `~/.local/share/ochat/extraction.log` for errors.)

- [ ] **Step 5: Smoke test 3 — confirm cross-thread retrieval**

Run: `ochat --thread other`, ask "What's my favorite color?" (a thread that has never discussed color before).
Expected: the reply correctly answers "teal" (or whatever was set), demonstrating the fact was retrieved from `memory.db` via semantic search rather than from thread history.

- [ ] **Step 6: Record the smoke test result**

If all three smoke tests pass, the implementation is complete. If any fails, return to the relevant task above — do not patch around it in Task 10.
