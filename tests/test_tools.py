import json
import sys
from io import BytesIO
from pathlib import Path
from unittest.mock import MagicMock, patch, call

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import ochat_tools


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def fresh_registry():
    ochat_tools.clear_registry()
    yield
    ochat_tools.clear_registry()


def _make_tool(name="dummy", dangerous=False):
    return ochat_tools.ToolDef(
        name=name,
        description="A test tool",
        parameters={
            "type": "object",
            "properties": {"x": {"type": "string"}},
            "required": ["x"],
        },
        fn=lambda x: f"result:{x}",
        dangerous=dangerous,
    )


def _mock_proc(*rpc_responses):
    """Fake subprocess whose stdout pre-loads JSON-RPC responses."""
    proc = MagicMock()
    proc.stdin = MagicMock()
    proc.stdout = BytesIO(
        b"".join(json.dumps(r).encode() + b"\n" for r in rpc_responses)
    )
    return proc


def _std_mcp_startup(tool_defs=None):
    """Standard pair of RPC responses for MCP server startup."""
    return (
        {"jsonrpc": "2.0", "id": 1, "result": {"protocolVersion": "2024-11-05", "capabilities": {}}},
        {"jsonrpc": "2.0", "id": 2, "result": {"tools": tool_defs or []}},
    )


# ---------------------------------------------------------------------------
# Tool registry
# ---------------------------------------------------------------------------

def test_register_and_retrieve_tool():
    t = _make_tool("foo")
    ochat_tools.register(t)
    assert ochat_tools.get_tool("foo") is t


def test_get_unknown_tool_returns_none():
    assert ochat_tools.get_tool("missing") is None


def test_all_tools_returns_all_registered():
    ochat_tools.register(_make_tool("a"))
    ochat_tools.register(_make_tool("b"))
    assert {t.name for t in ochat_tools.all_tools()} == {"a", "b"}


def test_clear_registry_removes_all_tools():
    ochat_tools.register(_make_tool("a"))
    ochat_tools.clear_registry()
    assert ochat_tools.all_tools() == []


def test_get_ollama_tools_returns_correct_schema():
    ochat_tools.register(_make_tool("greet"))
    schemas = ochat_tools.get_ollama_tools()
    assert len(schemas) == 1
    s = schemas[0]
    assert s["type"] == "function"
    assert s["function"]["name"] == "greet"
    assert "parameters" in s["function"]
    assert "description" in s["function"]


# ---------------------------------------------------------------------------
# Tool execution
# ---------------------------------------------------------------------------

def test_execute_tool_invokes_fn_with_arguments():
    calls = []
    ochat_tools.register(ochat_tools.ToolDef(
        name="log", description="", parameters={},
        fn=lambda msg: calls.append(msg) or f"logged:{msg}",
        dangerous=False,
    ))
    result = ochat_tools.execute_tool("log", {"msg": "hi"}, confirm_fn=None)
    assert result == "logged:hi"
    assert calls == ["hi"]


def test_execute_unknown_tool_returns_error_string():
    result = ochat_tools.execute_tool("no-such-tool", {}, confirm_fn=None)
    assert "unknown tool" in result.lower()


def test_execute_safe_tool_never_calls_confirm_fn():
    called = []
    ochat_tools.register(_make_tool("safe", dangerous=False))
    ochat_tools.execute_tool("safe", {"x": "v"}, confirm_fn=lambda n, a: called.append(True) or True)
    assert called == []


def test_execute_dangerous_tool_calls_confirm_fn_with_name_and_args():
    ochat_tools.register(_make_tool("risky", dangerous=True))
    seen = []
    ochat_tools.execute_tool("risky", {"x": "val"}, confirm_fn=lambda n, a: seen.append((n, a)) or True)
    assert seen == [("risky", {"x": "val"})]


def test_execute_dangerous_tool_returns_cancellation_message_when_denied():
    ochat_tools.register(_make_tool("risky", dangerous=True))
    result = ochat_tools.execute_tool("risky", {"x": "val"}, confirm_fn=lambda n, a: False)
    assert "cancel" in result.lower()


def test_execute_tool_returns_error_string_on_exception():
    ochat_tools.register(ochat_tools.ToolDef(
        name="broken", description="", parameters={},
        fn=lambda: 1 / 0,
        dangerous=False,
    ))
    result = ochat_tools.execute_tool("broken", {}, confirm_fn=None)
    assert "error" in result.lower()


# ---------------------------------------------------------------------------
# Built-in tools
# ---------------------------------------------------------------------------

def test_read_file_returns_file_contents(tmp_path):
    f = tmp_path / "note.txt"
    f.write_text("hello from file")
    assert ochat_tools._read_file(str(f)) == "hello from file"


def test_read_file_missing_path_returns_error(tmp_path):
    result = ochat_tools._read_file(str(tmp_path / "nope.txt"))
    assert "error" in result.lower() or "not found" in result.lower()


def test_write_file_writes_content_to_disk(tmp_path):
    p = tmp_path / "out.txt"
    result = ochat_tools._write_file(str(p), "written content")
    assert p.read_text() == "written content"
    assert "written" in result.lower()


def test_run_shell_returns_command_stdout():
    result = ochat_tools._run_shell("echo ochat_test_marker")
    assert "ochat_test_marker" in result


def test_run_shell_reports_nonzero_exit():
    result = ochat_tools._run_shell("false")  # always exits 1
    assert "exit" in result.lower() or "1" in result


def test_web_search_returns_abstract_text_from_ddg():
    fake_resp = MagicMock()
    fake_resp.json.return_value = {
        "AbstractText": "Manila is the capital city of the Philippines.",
        "RelatedTopics": [{"Text": "Metro Manila"}, {"Text": "Luzon"}],
    }
    fake_resp.raise_for_status = MagicMock()
    with patch("ochat_tools._requests.get", return_value=fake_resp):
        result = ochat_tools._web_search("Manila Philippines")
    assert "Manila" in result


def test_web_search_uses_brave_api_when_key_is_set():
    fake_resp = MagicMock()
    fake_resp.raise_for_status = MagicMock()
    fake_resp.json.return_value = {
        "web": {"results": [
            {"title": "2026 NBA Draft", "description": "Warriors selected player X at pick 11."},
            {"title": "NBA Draft results", "description": "Full list of picks from tonight."},
        ]}
    }
    with patch.dict("os.environ", {"BRAVE_API_KEY": "test-key-abc"}), \
         patch("ochat_tools._requests.get", return_value=fake_resp) as mock_get:
        result = ochat_tools._web_search("2026 NBA Draft Warriors")
    # Must call Brave endpoint with the key header
    call_args = mock_get.call_args
    assert "api.search.brave.com" in call_args.args[0]
    assert call_args.kwargs["headers"]["X-Subscription-Token"] == "test-key-abc"
    assert "Warriors" in result or "NBA Draft" in result


def test_web_search_skips_brave_when_no_key_set():
    fake_resp = MagicMock()
    fake_resp.raise_for_status = MagicMock()
    fake_resp.json.return_value = {"AbstractText": "DuckDuckGo result", "RelatedTopics": []}
    with patch.dict("os.environ", {}, clear=True), \
         patch("ochat_tools._requests.get", return_value=fake_resp) as mock_get:
        # Remove key if present in env
        import os
        os.environ.pop("BRAVE_API_KEY", None)
        result = ochat_tools._web_search("test query")
    call_args = mock_get.call_args
    assert "brave" not in call_args.args[0].lower()


def test_web_search_falls_back_to_html_results_when_instant_answer_empty():
    # DuckDuckGo's Instant Answer API returns no AbstractText for live events
    # (e.g. breaking sports news). The fallback must scrape HTML search results.
    empty_json = MagicMock()
    empty_json.raise_for_status = MagicMock()
    empty_json.json.return_value = {"AbstractText": "", "RelatedTopics": []}

    html_resp = MagicMock()
    html_resp.raise_for_status = MagicMock()
    html_resp.text = """
    <html><body>
      <a class="result__a" href="#">2026 NBA Draft Results</a>
      <a class="result__snippet" href="#">The 11th pick in the 2026 NBA Draft was announced live.</a>
      <a class="result__a" href="#">NBA Draft picks tonight</a>
      <a class="result__snippet" href="#">Full results from the 2026 NBA Draft ceremony.</a>
    </body></html>
    """

    with patch("ochat_tools._requests.get", side_effect=[empty_json, html_resp]):
        result = ochat_tools._web_search("2026 NBA Draft pick 11")
    assert "2026 NBA Draft" in result or "pick" in result.lower()


def test_web_search_returns_error_string_on_network_failure():
    with patch("ochat_tools._requests.get", side_effect=Exception("timeout")):
        result = ochat_tools._web_search("anything")
    assert "failed" in result.lower() or "error" in result.lower()


def test_builtin_tools_dict_has_expected_names():
    assert set(ochat_tools.BUILTIN_TOOLS) == {"web_search", "read_file", "write_file", "run_shell"}


def test_write_file_and_run_shell_are_marked_dangerous():
    assert ochat_tools.BUILTIN_TOOLS["write_file"].dangerous is True
    assert ochat_tools.BUILTIN_TOOLS["run_shell"].dangerous is True


def test_web_search_and_read_file_are_not_dangerous():
    assert ochat_tools.BUILTIN_TOOLS["web_search"].dangerous is False
    assert ochat_tools.BUILTIN_TOOLS["read_file"].dangerous is False


# ---------------------------------------------------------------------------
# MCP client
# ---------------------------------------------------------------------------

def test_mcp_client_sends_initialize_on_start():
    proc = _mock_proc(*_std_mcp_startup())
    with patch("subprocess.Popen", return_value=proc):
        client = ochat_tools.MCPClient("srv", ["fake-server"])
        client.start()

    written = b"".join(c.args[0] for c in proc.stdin.write.call_args_list)
    msgs = [json.loads(line) for line in written.decode().splitlines() if line.strip()]
    assert any(m.get("method") == "initialize" for m in msgs)


def test_mcp_client_sends_tools_list_on_start():
    proc = _mock_proc(*_std_mcp_startup())
    with patch("subprocess.Popen", return_value=proc):
        client = ochat_tools.MCPClient("srv", ["fake-server"])
        client.start()

    written = b"".join(c.args[0] for c in proc.stdin.write.call_args_list)
    msgs = [json.loads(line) for line in written.decode().splitlines() if line.strip()]
    assert any(m.get("method") == "tools/list" for m in msgs)


def test_mcp_client_registers_server_tools_in_registry():
    tool_defs = [{"name": "search", "description": "Search docs",
                  "inputSchema": {"type": "object", "properties": {"q": {"type": "string"}}, "required": ["q"]}}]
    proc = _mock_proc(*_std_mcp_startup(tool_defs))
    with patch("subprocess.Popen", return_value=proc):
        client = ochat_tools.MCPClient("docs", ["fake-server"])
        client.start()

    assert ochat_tools.get_tool("docs__search") is not None


def test_mcp_client_namespaces_tools_with_server_name():
    tool_defs = [{"name": "lookup", "description": "", "inputSchema": {"type": "object", "properties": {}}}]
    proc = _mock_proc(*_std_mcp_startup(tool_defs))
    with patch("subprocess.Popen", return_value=proc):
        client = ochat_tools.MCPClient("myserver", ["fake"])
        client.start()

    assert ochat_tools.get_tool("myserver__lookup") is not None
    assert ochat_tools.get_tool("lookup") is None  # unnamespaced form must NOT be registered


def test_mcp_client_call_tool_returns_text_content():
    call_resp = {"jsonrpc": "2.0", "id": 3,
                 "result": {"content": [{"type": "text", "text": "search result here"}]}}
    proc = _mock_proc(*_std_mcp_startup(), call_resp)
    with patch("subprocess.Popen", return_value=proc):
        client = ochat_tools.MCPClient("srv", ["fake"])
        client.start()
        result = client.call_tool("mytool", {"key": "val"})

    assert result == "search result here"


def test_mcp_client_stop_terminates_subprocess():
    proc = _mock_proc(*_std_mcp_startup())
    with patch("subprocess.Popen", return_value=proc):
        client = ochat_tools.MCPClient("srv", ["fake"])
        client.start()
        client.stop()

    proc.terminate.assert_called_once()


def test_mcp_client_exposes_server_tools_list():
    tool_defs = [{"name": "t1", "description": "tool 1", "inputSchema": {}}]
    proc = _mock_proc(*_std_mcp_startup(tool_defs))
    with patch("subprocess.Popen", return_value=proc):
        client = ochat_tools.MCPClient("srv", ["fake"])
        client.start()

    assert len(client.server_tools) == 1
    assert client.server_tools[0]["name"] == "t1"


# ---------------------------------------------------------------------------
# Tool execution loop
# ---------------------------------------------------------------------------

def test_run_tool_loop_returns_text_when_model_makes_no_tool_calls():
    def chat_fn(messages, tools):
        return "plain answer", None

    result = ochat_tools.run_tool_loop(
        [{"role": "user", "content": "hi"}], chat_fn, confirm_fn=None
    )
    assert result == "plain answer"


def test_run_tool_loop_executes_tool_and_sends_result_back():
    ochat_tools.register(ochat_tools.ToolDef(
        name="add", description="", parameters={},
        fn=lambda a, b: str(int(a) + int(b)),
        dangerous=False,
    ))
    call_count = [0]

    def chat_fn(messages, tools):
        call_count[0] += 1
        if call_count[0] == 1:
            return "", [{"function": {"name": "add", "arguments": {"a": "2", "b": "3"}}}]
        return "answer is 5", None

    result = ochat_tools.run_tool_loop(
        [{"role": "user", "content": "2+3?"}], chat_fn, confirm_fn=None
    )
    assert result == "answer is 5"
    assert call_count[0] == 2


def test_run_tool_loop_appends_tool_result_to_messages():
    ochat_tools.register(ochat_tools.ToolDef(
        name="ping", description="", parameters={},
        fn=lambda: "pong",
        dangerous=False,
    ))
    received = []
    call_count = [0]

    def chat_fn(messages, tools):
        call_count[0] += 1
        if call_count[0] == 1:
            return "", [{"function": {"name": "ping", "arguments": {}}}]
        received.extend(messages)
        return "done", None

    ochat_tools.run_tool_loop([{"role": "user", "content": "ping?"}], chat_fn, confirm_fn=None)
    tool_msgs = [m for m in received if m.get("role") == "tool"]
    assert len(tool_msgs) == 1
    assert tool_msgs[0]["content"] == "pong"


def test_run_tool_loop_handles_string_encoded_arguments():
    """Ollama sometimes returns arguments as a JSON string rather than a dict."""
    ochat_tools.register(ochat_tools.ToolDef(
        name="echo", description="", parameters={},
        fn=lambda text: f"echo:{text}",
        dangerous=False,
    ))
    call_count = [0]

    def chat_fn(messages, tools):
        call_count[0] += 1
        if call_count[0] == 1:
            return "", [{"function": {"name": "echo", "arguments": '{"text": "hello"}'}}]
        return "done", None

    result = ochat_tools.run_tool_loop([{"role": "user", "content": "echo"}], chat_fn, confirm_fn=None)
    assert result == "done"


def test_run_tool_loop_stops_after_max_turns_and_makes_final_call():
    # Registry must be non-empty so get_ollama_tools() returns tools and the loop runs.
    ochat_tools.register(_make_tool("placeholder"))
    call_count = [0]

    def chat_fn(messages, tools):
        call_count[0] += 1
        if tools:
            return "", [{"function": {"name": "unknown_tool", "arguments": {}}}]
        return "gave up", None

    result = ochat_tools.run_tool_loop(
        [{"role": "user", "content": "loop"}], chat_fn, confirm_fn=None, max_turns=3
    )
    assert result == "gave up"
    assert call_count[0] == 4  # 3 tool turns + 1 final call with empty tools


# ---------------------------------------------------------------------------
# Config and init
# ---------------------------------------------------------------------------

def test_load_config_returns_empty_dict_when_file_missing(tmp_path):
    assert ochat_tools.load_config(tmp_path / "nope.json") == {}


def test_load_config_reads_json_file(tmp_path):
    cfg = tmp_path / "tools.json"
    cfg.write_text(json.dumps({"enabled": True, "builtin": {"web_search": True}}))
    result = ochat_tools.load_config(cfg)
    assert result["enabled"] is True
    assert result["builtin"]["web_search"] is True


def test_init_tools_returns_empty_and_registers_nothing_when_disabled():
    clients = ochat_tools.init_tools(config={"enabled": False})
    assert clients == []
    assert ochat_tools.all_tools() == []


def test_init_tools_registers_only_enabled_builtins():
    ochat_tools.init_tools(config={
        "enabled": True,
        "builtin": {"web_search": True, "read_file": True, "write_file": False, "run_shell": False},
        "mcp_servers": [],
    })
    names = {t.name for t in ochat_tools.all_tools()}
    assert "web_search" in names
    assert "read_file" in names
    assert "write_file" not in names
    assert "run_shell" not in names


def test_init_tools_registers_all_builtins_when_builtin_key_absent():
    ochat_tools.init_tools(config={"enabled": True, "mcp_servers": []})
    names = {t.name for t in ochat_tools.all_tools()}
    assert names >= {"web_search", "read_file", "write_file", "run_shell"}


def test_init_tools_starts_mcp_server_and_returns_client():
    proc = _mock_proc(*_std_mcp_startup())
    with patch("subprocess.Popen", return_value=proc):
        clients = ochat_tools.init_tools(config={
            "enabled": True,
            "builtin": {},
            "mcp_servers": [{"name": "test-srv", "command": ["fake-cmd"]}],
        })
    assert len(clients) == 1
    assert clients[0].name == "test-srv"


def test_init_tools_tolerates_failed_mcp_server_startup(capsys):
    with patch("subprocess.Popen", side_effect=FileNotFoundError("no such cmd")):
        clients = ochat_tools.init_tools(config={
            "enabled": True,
            "builtin": {},
            "mcp_servers": [{"name": "bad-srv", "command": ["nonexistent-cmd"]}],
        })
    assert clients == []
    assert "bad-srv" in capsys.readouterr().err


def test_shutdown_tools_stops_all_active_clients():
    proc = _mock_proc(*_std_mcp_startup())
    with patch("subprocess.Popen", return_value=proc):
        ochat_tools.init_tools(config={
            "enabled": True, "builtin": {},
            "mcp_servers": [{"name": "s", "command": ["fake"]}],
        })
    ochat_tools.shutdown_tools()
    proc.terminate.assert_called_once()


def test_init_tools_registers_web_search_by_default_when_no_config_file(tmp_path):
    # web_search should work out of the box — no config file required.
    # Dangerous tools (write_file, run_shell) must not be auto-registered.
    clients = ochat_tools.init_tools(config_path=tmp_path / "nonexistent.json")
    assert clients == []
    names = {t.name for t in ochat_tools.all_tools()}
    assert "web_search" in names
    assert "write_file" not in names
    assert "run_shell" not in names


def test_init_tools_with_enabled_false_disables_even_default_web_search():
    # Explicit opt-out via config must suppress the default web_search too.
    ochat_tools.init_tools(config={"enabled": False})
    assert ochat_tools.all_tools() == []
