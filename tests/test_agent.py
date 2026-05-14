"""Tests for agent/agent.py – uses mocks to avoid real LLM calls."""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from agent.agent import CodingAgent, _fmt_args


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_message(content: str | None = None, tool_calls: list | None = None) -> MagicMock:
    """Return a mock ChatCompletionMessage."""
    msg = MagicMock()
    msg.content = content
    msg.tool_calls = tool_calls or []
    # model_dump should return a plain dict the agent can append to messages
    dump: dict[str, Any] = {"role": "assistant"}
    if content:
        dump["content"] = content
    if tool_calls:
        dump["tool_calls"] = tool_calls
    msg.model_dump.return_value = dump
    return msg


def _make_tool_call(call_id: str, name: str, arguments: dict) -> MagicMock:
    tc = MagicMock()
    tc.id = call_id
    tc.function.name = name
    tc.function.arguments = json.dumps(arguments)
    return tc


def _make_response(message: MagicMock) -> MagicMock:
    resp = MagicMock()
    resp.choices = [MagicMock(message=message)]
    return resp


# ---------------------------------------------------------------------------
# CodingAgent.run
# ---------------------------------------------------------------------------


class TestCodingAgentRun:
    @patch("agent.agent.OpenAI")
    def test_returns_final_message_when_no_tool_calls(self, mock_openai_cls) -> None:
        client = MagicMock()
        mock_openai_cls.return_value = client

        client.chat.completions.create.return_value = _make_response(
            _make_message(content="Task complete!")
        )

        agent = CodingAgent(api_key="test-key", verbose=False)
        result = agent.run("do something")
        assert result == "Task complete!"

    @patch("agent.agent.OpenAI")
    def test_executes_tool_call_then_finishes(self, mock_openai_cls, tmp_path) -> None:
        client = MagicMock()
        mock_openai_cls.return_value = client

        tool_call = _make_tool_call("call_1", "list_directory", {"path": str(tmp_path)})
        first_response = _make_response(_make_message(tool_calls=[tool_call]))
        second_response = _make_response(_make_message(content="Listed the directory."))

        client.chat.completions.create.side_effect = [first_response, second_response]

        agent = CodingAgent(api_key="test-key", verbose=False)
        result = agent.run("list the tmp directory")

        assert result == "Listed the directory."
        assert client.chat.completions.create.call_count == 2

    @patch("agent.agent.OpenAI")
    def test_stops_after_max_iterations(self, mock_openai_cls) -> None:
        client = MagicMock()
        mock_openai_cls.return_value = client

        # Always return a tool call so the agent never finishes naturally
        def always_tool_call(*args, **kwargs):
            tool_call = _make_tool_call("call_x", "run_command", {"command": "echo hi"})
            return _make_response(_make_message(tool_calls=[tool_call]))

        client.chat.completions.create.side_effect = always_tool_call

        agent = CodingAgent(api_key="test-key", max_iterations=3, verbose=False)
        result = agent.run("infinite loop task")

        assert "maximum number of iterations" in result
        assert client.chat.completions.create.call_count == 3

    @patch("agent.agent.OpenAI")
    def test_tool_result_appended_to_messages(self, mock_openai_cls, tmp_path) -> None:
        client = MagicMock()
        mock_openai_cls.return_value = client

        f = tmp_path / "hello.txt"
        f.write_text("file content", encoding="utf-8")

        tool_call = _make_tool_call("call_read", "read_file", {"path": str(f)})
        first_response = _make_response(_make_message(tool_calls=[tool_call]))
        second_response = _make_response(_make_message(content="Done reading."))

        client.chat.completions.create.side_effect = [first_response, second_response]

        agent = CodingAgent(api_key="test-key", verbose=False)
        agent.run("read a file")

        # Second call messages should include the tool result
        second_call_messages = client.chat.completions.create.call_args_list[1][1]["messages"]
        tool_result_msgs = [m for m in second_call_messages if m.get("role") == "tool"]
        assert len(tool_result_msgs) == 1
        assert tool_result_msgs[0]["content"] == "file content"

    @patch("agent.agent.OpenAI")
    def test_invalid_json_arguments_handled_gracefully(self, mock_openai_cls) -> None:
        client = MagicMock()
        mock_openai_cls.return_value = client

        tool_call = MagicMock()
        tool_call.id = "bad_json"
        tool_call.function.name = "run_command"
        tool_call.function.arguments = "not valid json {{{"

        first_response = _make_response(_make_message(tool_calls=[tool_call]))
        second_response = _make_response(_make_message(content="Handled."))
        client.chat.completions.create.side_effect = [first_response, second_response]

        agent = CodingAgent(api_key="test-key", verbose=False)
        # Should not raise, even with bad JSON args
        result = agent.run("do something")
        assert result == "Handled."

    @patch("agent.agent.OpenAI")
    def test_stream_yields_characters(self, mock_openai_cls) -> None:
        client = MagicMock()
        mock_openai_cls.return_value = client

        client.chat.completions.create.return_value = _make_response(
            _make_message(content="abc")
        )

        agent = CodingAgent(api_key="test-key", verbose=False)
        chunks = list(agent.stream("task"))
        assert "".join(chunks) == "abc"

    @patch("agent.agent.OpenAI")
    def test_empty_content_returns_empty_string(self, mock_openai_cls) -> None:
        client = MagicMock()
        mock_openai_cls.return_value = client

        client.chat.completions.create.return_value = _make_response(
            _make_message(content=None)
        )

        agent = CodingAgent(api_key="test-key", verbose=False)
        result = agent.run("task")
        assert result == ""


# ---------------------------------------------------------------------------
# _fmt_args helper
# ---------------------------------------------------------------------------


class TestFmtArgs:
    def test_simple_string(self) -> None:
        result = _fmt_args({"path": "/tmp/foo.txt"})
        assert 'path="/tmp/foo.txt"' in result

    def test_long_string_truncated(self) -> None:
        long_val = "x" * 100
        result = _fmt_args({"content": long_val})
        assert "..." in result
        assert len(result) < 100  # much shorter than original

    def test_non_string_value_uses_repr(self) -> None:
        result = _fmt_args({"count": 42})
        assert "count=42" in result

    def test_multiple_args(self) -> None:
        result = _fmt_args({"path": "f.txt", "content": "hello"})
        assert "path=" in result
        assert "content=" in result
