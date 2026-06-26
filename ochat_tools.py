"""
ochat_tools.py — tool calling, MCP client, and built-in tools for ochat.

Loaded by ochat.py when ~/.config/ochat/tools.json exists and has "enabled": true.
ochat_tools never imports ochat; the integration lives entirely in ochat.py.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import requests as _requests

# ---------------------------------------------------------------------------
# Tool definition and registry
# ---------------------------------------------------------------------------

@dataclass
class ToolDef:
    name: str
    description: str
    parameters: dict  # JSON Schema object
    fn: Callable[..., str]
    dangerous: bool = False  # requires user confirmation before execution


_registry: dict[str, ToolDef] = {}
_registry_lock = threading.Lock()


def register(tool: ToolDef) -> None:
    with _registry_lock:
        _registry[tool.name] = tool


def unregister(name: str) -> None:
    with _registry_lock:
        _registry.pop(name, None)


def clear_registry() -> None:
    with _registry_lock:
        _registry.clear()


def get_tool(name: str) -> ToolDef | None:
    return _registry.get(name)


def all_tools() -> list[ToolDef]:
    return list(_registry.values())


def get_ollama_tools() -> list[dict]:
    """Return tool definitions in Ollama's tools array format."""
    return [
        {
            "type": "function",
            "function": {
                "name": t.name,
                "description": t.description,
                "parameters": t.parameters,
            },
        }
        for t in _registry.values()
    ]


# ---------------------------------------------------------------------------
# Tool execution
# ---------------------------------------------------------------------------

def terminal_confirm(tool_name: str, arguments: dict) -> bool:
    args_str = json.dumps(arguments, indent=2)
    print(f"\n[tool: {tool_name}] arguments:\n{args_str}")
    try:
        answer = input("Allow? [y/N] ").strip().lower()
    except EOFError:
        return False
    return answer in ("y", "yes")


def execute_tool(
    name: str,
    arguments: dict,
    confirm_fn: Callable[[str, dict], bool] | None = terminal_confirm,
) -> str:
    tool = _registry.get(name)
    if tool is None:
        return f"Error: unknown tool '{name}'"
    if tool.dangerous and confirm_fn is not None:
        if not confirm_fn(name, arguments):
            return "Cancelled by user."
    try:
        return tool.fn(**arguments)
    except TypeError as exc:
        return f"Error: wrong arguments for '{name}': {exc}"
    except Exception as exc:
        return f"Error executing '{name}': {exc}"


# ---------------------------------------------------------------------------
# Built-in tools
# ---------------------------------------------------------------------------

def _web_search(query: str) -> str:
    import html as _html
    import re as _re

    # Try the Instant Answer API first — fast and structured for well-known facts.
    try:
        resp = _requests.get(
            "https://api.duckduckgo.com/",
            params={"q": query, "format": "json", "no_html": "1", "skip_disambig": "1"},
            timeout=10,
        )
        resp.raise_for_status()
        data = resp.json()
        parts: list[str] = []
        if data.get("AbstractText"):
            parts.append(data["AbstractText"])
        for topic in data.get("RelatedTopics", [])[:5]:
            if isinstance(topic, dict) and topic.get("Text"):
                parts.append(topic["Text"])
        if parts:
            return "\n".join(parts)
    except Exception:
        pass

    # Fallback: scrape the HTML search page for real-time results (live events, news).
    try:
        headers = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"}
        resp = _requests.get(
            "https://html.duckduckgo.com/html/",
            params={"q": query},
            headers=headers,
            timeout=10,
        )
        resp.raise_for_status()
        text = resp.text
        titles = _re.findall(r'class="result__a"[^>]*>(.*?)</a>', text, _re.DOTALL)
        snippets = _re.findall(r'class="result__snippet"[^>]*>(.*?)</a>', text, _re.DOTALL)
        results: list[str] = []
        for title, snippet in zip(titles[:5], snippets[:5]):
            clean_t = _html.unescape(_re.sub(r"<[^>]+>", "", title).strip())
            clean_s = _html.unescape(_re.sub(r"<[^>]+>", "", snippet).strip())
            if clean_t or clean_s:
                results.append(f"{clean_t}: {clean_s}" if clean_t and clean_s else clean_t or clean_s)
        return "\n".join(results) if results else "No results found."
    except Exception as exc:
        return f"Web search failed: {exc}"


def _read_file(path: str) -> str:
    try:
        p = Path(path).expanduser()
        if not p.exists():
            return f"Error: file not found: {path}"
        if p.stat().st_size > 1_000_000:
            return f"Error: file too large (>{1_000_000} bytes): {path}"
        return p.read_text(encoding="utf-8", errors="replace")
    except Exception as exc:
        return f"Error reading file: {exc}"


def _write_file(path: str, content: str) -> str:
    try:
        p = Path(path).expanduser()
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content, encoding="utf-8")
        return f"Written {len(content)} characters to {path}"
    except Exception as exc:
        return f"Error writing file: {exc}"


def _run_shell(command: str) -> str:
    try:
        result = subprocess.run(
            command, shell=True, capture_output=True, text=True, timeout=30
        )
        out = result.stdout.strip()
        err = result.stderr.strip()
        if result.returncode != 0:
            return f"Exit {result.returncode}\n{err or out}"
        return out or "(no output)"
    except subprocess.TimeoutExpired:
        return "Error: command timed out after 30 seconds"
    except Exception as exc:
        return f"Error running shell command: {exc}"


BUILTIN_TOOLS: dict[str, ToolDef] = {
    "web_search": ToolDef(
        name="web_search",
        description="Search the web for current information using DuckDuckGo.",
        parameters={
            "type": "object",
            "properties": {"query": {"type": "string", "description": "The search query"}},
            "required": ["query"],
        },
        fn=_web_search,
        dangerous=False,
    ),
    "read_file": ToolDef(
        name="read_file",
        description="Read the contents of a file from the local filesystem.",
        parameters={
            "type": "object",
            "properties": {"path": {"type": "string", "description": "Absolute or ~ path to the file"}},
            "required": ["path"],
        },
        fn=_read_file,
        dangerous=False,
    ),
    "write_file": ToolDef(
        name="write_file",
        description="Write content to a file on the local filesystem.",
        parameters={
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Absolute or ~ path to write"},
                "content": {"type": "string", "description": "Content to write"},
            },
            "required": ["path", "content"],
        },
        fn=_write_file,
        dangerous=True,
    ),
    "run_shell": ToolDef(
        name="run_shell",
        description="Run a shell command and return its output.",
        parameters={
            "type": "object",
            "properties": {"command": {"type": "string", "description": "The shell command to run"}},
            "required": ["command"],
        },
        fn=_run_shell,
        dangerous=True,
    ),
}


# ---------------------------------------------------------------------------
# MCP client (stdio transport, JSON-RPC 2.0)
# ---------------------------------------------------------------------------

class MCPClient:
    def __init__(self, name: str, command: list[str], env: dict[str, str] | None = None):
        self.name = name
        self.command = command
        self.env = env or {}
        self._proc: subprocess.Popen | None = None
        self._lock = threading.Lock()
        self._next_id = 1
        self._server_tools: list[dict] = []

    def start(self) -> None:
        proc_env = os.environ.copy()
        proc_env.update(self.env)
        self._proc = subprocess.Popen(
            self.command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            env=proc_env,
        )
        self._rpc("initialize", {
            "protocolVersion": "2024-11-05",
            "capabilities": {"tools": {}},
            "clientInfo": {"name": "ochat", "version": "1.0"},
        })
        self._notify("notifications/initialized")
        result = self._rpc("tools/list")
        self._server_tools = result.get("tools", [])
        self._register_server_tools()

    def stop(self) -> None:
        if self._proc:
            self._proc.terminate()
            try:
                self._proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self._proc.kill()
            self._proc = None

    def call_tool(self, tool_name: str, arguments: dict) -> str:
        result = self._rpc("tools/call", {"name": tool_name, "arguments": arguments})
        content = result.get("content", [])
        parts = [item["text"] for item in content if item.get("type") == "text"]
        return "\n".join(parts) if parts else str(result)

    @property
    def server_tools(self) -> list[dict]:
        return list(self._server_tools)

    def _rpc(self, method: str, params: dict | None = None) -> Any:
        with self._lock:
            msg_id = self._next_id
            self._next_id += 1
            request: dict = {"jsonrpc": "2.0", "id": msg_id, "method": method}
            if params is not None:
                request["params"] = params
            self._write(request)
            while True:
                raw = self._proc.stdout.readline()
                if not raw:
                    raise RuntimeError(f"MCP server '{self.name}' closed unexpectedly")
                try:
                    msg = json.loads(raw)
                except json.JSONDecodeError:
                    continue
                if msg.get("id") == msg_id:
                    if "error" in msg:
                        raise RuntimeError(f"MCP '{self.name}' error: {msg['error']}")
                    return msg.get("result", {})

    def _notify(self, method: str, params: dict | None = None) -> None:
        notification: dict = {"jsonrpc": "2.0", "method": method}
        if params is not None:
            notification["params"] = params
        self._write(notification)

    def _write(self, obj: dict) -> None:
        line = json.dumps(obj) + "\n"
        self._proc.stdin.write(line.encode())
        self._proc.stdin.flush()

    def _register_server_tools(self) -> None:
        for tool_def in self._server_tools:
            tool_name = tool_def["name"]
            namespaced = f"{self.name}__{tool_name}"

            def make_fn(tn: str = tool_name) -> Callable[..., str]:
                def fn(**kwargs: Any) -> str:
                    return self.call_tool(tn, kwargs)
                return fn

            register(ToolDef(
                name=namespaced,
                description=f"[{self.name}] {tool_def.get('description', '')}",
                parameters=tool_def.get("inputSchema", {"type": "object", "properties": {}}),
                fn=make_fn(),
                dangerous=False,
            ))


# ---------------------------------------------------------------------------
# Tool execution loop
# ---------------------------------------------------------------------------

def run_tool_loop(
    messages: list[dict],
    chat_fn: Callable[[list[dict], list[dict]], tuple[str, list[dict] | None]],
    confirm_fn: Callable[[str, dict], bool] | None = terminal_confirm,
    max_turns: int = 10,
) -> str:
    """
    Agentic loop: call chat_fn(messages, tools) → (text, tool_calls | None).
    Execute any tool_calls, append results, repeat until plain-text reply or
    max_turns exhausted (then calls chat_fn once more with empty tools list).
    """
    tools = get_ollama_tools()
    current_messages = list(messages)

    for _ in range(max_turns):
        text, tool_calls = chat_fn(current_messages, tools)
        if not tool_calls:
            return text

        current_messages.append({
            "role": "assistant",
            "content": text or "",
            "tool_calls": tool_calls,
        })

        for tc in tool_calls:
            fn_name = tc["function"]["name"]
            raw_args = tc["function"].get("arguments", {})
            if isinstance(raw_args, str):
                try:
                    raw_args = json.loads(raw_args)
                except json.JSONDecodeError:
                    raw_args = {}
            print(f"\n[tool: {fn_name}]", end=" ", flush=True)
            result = execute_tool(fn_name, raw_args, confirm_fn=confirm_fn)
            print(f"done", flush=True)
            current_messages.append({"role": "tool", "content": result})

    # Max turns reached — final call without tools so model can summarise
    text, _ = chat_fn(current_messages, [])
    return text


# ---------------------------------------------------------------------------
# Config loading and initialisation
# ---------------------------------------------------------------------------

DEFAULT_CONFIG_PATH = Path.home() / ".config" / "ochat" / "tools.json"

_active_clients: list[MCPClient] = []


def load_config(path: Path = DEFAULT_CONFIG_PATH) -> dict:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}


def init_tools(
    config: dict | None = None,
    config_path: Path = DEFAULT_CONFIG_PATH,
) -> list[MCPClient]:
    """
    Initialise the tool registry from config. Returns started MCP clients.
    Call shutdown_tools() on exit to terminate them.

    With no config file (or config={}), registers web_search as a safe
    default — no dangerous tools, no MCP servers. Set "enabled": false in
    the config to suppress even the default web_search.
    """
    global _active_clients
    if config is None:
        config = load_config(config_path)

    clear_registry()

    if not config:
        # No config file — register web_search as a safe always-on default.
        register(BUILTIN_TOOLS["web_search"])
        _active_clients = []
        return []

    if not config.get("enabled", False):
        _active_clients = []
        return []

    builtin_cfg = config.get("builtin", None)
    for name, tool in BUILTIN_TOOLS.items():
        if builtin_cfg is None or builtin_cfg.get(name, True):
            register(tool)

    clients: list[MCPClient] = []
    for server_cfg in config.get("mcp_servers", []):
        try:
            client = MCPClient(
                name=server_cfg["name"],
                command=server_cfg["command"],
                env=server_cfg.get("env"),
            )
            client.start()
            clients.append(client)
        except Exception as exc:
            print(
                f"warning: failed to start MCP server '{server_cfg.get('name')}': {exc}",
                file=sys.stderr,
            )

    _active_clients = clients
    return clients


def shutdown_tools() -> None:
    global _active_clients
    for client in _active_clients:
        try:
            client.stop()
        except Exception:
            pass
    _active_clients = []
