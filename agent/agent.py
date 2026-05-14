"""Core agent loop for the AI coding agent."""

from __future__ import annotations

import json
import sys
from typing import Any, Iterator

from openai import OpenAI

from .config import (
    MAX_ITERATIONS,
    MODEL,
    OPENAI_API_KEY,
    OPENAI_BASE_URL,
    SYSTEM_PROMPT,
)
from .tools import TOOLS, execute_tool


class CodingAgent:
    """An AI coding agent that uses an LLM with tool-calling to accomplish tasks.

    Parameters
    ----------
    api_key:
        OpenAI (or compatible) API key. Falls back to the ``OPENAI_API_KEY``
        environment variable when not provided.
    base_url:
        Base URL for the API endpoint. Falls back to ``OPENAI_BASE_URL`` or the
        official OpenAI endpoint.
    model:
        Model name to use for completions. Falls back to ``AGENT_MODEL`` or
        ``gpt-4o``.
    max_iterations:
        Maximum number of LLM↔tool iterations before giving up.
    verbose:
        When *True*, print each tool call and its result to stdout.
    """

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        model: str | None = None,
        max_iterations: int | None = None,
        verbose: bool = True,
    ) -> None:
        self.model = model or MODEL
        self.max_iterations = max_iterations if max_iterations is not None else MAX_ITERATIONS
        self.verbose = verbose
        self._client = OpenAI(
            api_key=api_key or OPENAI_API_KEY or "sk-placeholder",
            base_url=base_url or OPENAI_BASE_URL,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self, task: str) -> str:
        """Run the agent on *task* and return the final response text.

        The agent iterates, calling tools as needed, until the LLM produces a
        final message without tool calls or ``max_iterations`` is reached.
        """
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": task},
        ]

        for iteration in range(1, self.max_iterations + 1):
            if self.verbose:
                print(f"\n[Agent] Iteration {iteration}/{self.max_iterations}", file=sys.stderr)

            response = self._client.chat.completions.create(
                model=self.model,
                messages=messages,
                tools=TOOLS,
                tool_choice="auto",
            )

            message = response.choices[0].message
            messages.append(message.model_dump(exclude_unset=True))

            # No tool calls → the agent is done
            if not message.tool_calls:
                return message.content or ""

            # Execute each tool call and feed results back
            for tool_call in message.tool_calls:
                name = tool_call.function.name
                try:
                    arguments = json.loads(tool_call.function.arguments)
                except json.JSONDecodeError:
                    arguments = {}

                if self.verbose:
                    print(f"[Tool] {name}({_fmt_args(arguments)})", file=sys.stderr)

                result = execute_tool(name, arguments)

                if self.verbose:
                    preview = result[:200] + "..." if len(result) > 200 else result
                    print(f"[Tool result] {preview}", file=sys.stderr)

                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": result,
                    }
                )

        return (
            "Agent stopped: maximum number of iterations reached "
            f"({self.max_iterations}). The task may be incomplete."
        )

    def stream(self, task: str) -> Iterator[str]:
        """Yield response text chunks while running *task*.

        This is a convenience wrapper around :meth:`run` that yields the final
        response character-by-character. Tool call output is still printed to
        stderr when ``verbose=True``.
        """
        result = self.run(task)
        yield from result


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fmt_args(args: dict[str, Any]) -> str:
    """Return a short, readable representation of tool arguments."""
    parts = []
    for k, v in args.items():
        v_str = repr(v) if not isinstance(v, str) else f'"{v}"'
        if len(v_str) > 60:
            v_str = v_str[:57] + '..."'
        parts.append(f"{k}={v_str}")
    return ", ".join(parts)
