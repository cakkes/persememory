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


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0.0:
        return 0.0
    return float(np.dot(a, b) / denom)


def top_k_facts(query_embedding, facts, k=RETRIEVAL_TOP_K, min_similarity=RETRIEVAL_MIN_SIMILARITY):
    scored = []
    for fact in facts:
        similarity = cosine_similarity(query_embedding, fact["embedding"])
        if similarity >= min_similarity:
            scored.append((similarity, fact))
    scored.sort(key=lambda pair: pair[0], reverse=True)
    return [fact for _, fact in scored[:k]]


def is_duplicate_fact(candidate_embedding, existing_embeddings, threshold=DEDUP_SIMILARITY_THRESHOLD):
    for embedding in existing_embeddings:
        if cosine_similarity(candidate_embedding, embedding) > threshold:
            return True
    return False


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


def current_datetime_context() -> str:
    now = datetime.now().astimezone()
    return f"Current date/time: {now.strftime('%A, %B %d, %Y, %I:%M %p %Z')}"


CALENDAR_KEYWORDS = (
    "calendar", "schedule", "scheduled", "meeting", "appointment", "event",
    "remind", "reminder", "monday", "tuesday", "wednesday", "thursday",
    "friday", "saturday", "sunday", "tomorrow", "tonight",
)


def looks_calendar_related(text: str) -> bool:
    lowered = text.lower()
    return any(keyword in lowered for keyword in CALENDAR_KEYWORDS)


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


def think_param(level: str):
    if level == "off":
        return False
    if level in ("on", "true"):
        return True
    return level


def _model_installed(required: str, installed_names: set[str]) -> bool:
    """Check whether a required model name is satisfied by the installed set.

    Ollama reports installed models with an explicit tag (e.g. pulling
    "nomic-embed-text" with no tag gets reported back as
    "nomic-embed-text:latest"). A required name with no tag should match
    any tagged variant of that same base name; a required name that already
    specifies a tag must match that exact tag.
    """
    if required in installed_names:
        return True
    if ":" not in required:
        return any(name.startswith(f"{required}:") for name in installed_names)
    return False


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
    missing = [model for model in (CHAT_MODEL, EMBED_MODEL) if not _model_installed(model, installed)]
    for model in missing:
        print(
            f"error: required model '{model}' is not installed — run `ollama pull {model}`",
            file=sys.stderr,
        )
    if missing:
        sys.exit(1)


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


EXTRACTION_PROMPT = (
    "Given this exchange, list any new durable facts or preferences about the "
    "user worth remembering long-term. Respond with ONLY a JSON array of short "
    'fact strings, e.g. ["prefers terse answers"]. If nothing is worth '
    "remembering, respond with []. Resolve any relative dates mentioned (e.g. "
    "'next Thursday') to absolute dates before recording a fact, using the "
    "current date/time context given."
)


def log_extraction_error(exc: Exception) -> None:
    EXTRACTION_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(EXTRACTION_LOG_PATH, "a", encoding="utf-8") as f:
        f.write(f"{datetime.now(timezone.utc).isoformat()} {exc!r}\n")


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


def extract_facts(conn, user_message: str, assistant_message: str, source_thread: str) -> None:
    try:
        exchange = f"User: {user_message}\nAssistant: {assistant_message}"
        reply = ollama_chat(
            [
                {"role": "system", "content": f"{EXTRACTION_PROMPT}\n\n{current_datetime_context()}"},
                {"role": "user", "content": exchange},
            ],
            think=False,
            stream_to_stdout=False,
        )
        candidate_facts = json.loads(_extract_json_array(reply))
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


def handle_turn(conn, thread, path, user_input, think):
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

    try:
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
