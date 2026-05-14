"""Tool definitions and implementations for the AI coding agent."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path
from typing import Any

from .config import MAX_OUTPUT_CHARS


# ---------------------------------------------------------------------------
# Tool implementations
# ---------------------------------------------------------------------------

def read_file(path: str) -> str:
    """Return the contents of *path*."""
    try:
        return Path(path).read_text(encoding="utf-8")
    except FileNotFoundError:
        return f"Error: file not found: {path}"
    except PermissionError:
        return f"Error: permission denied: {path}"
    except Exception as exc:  # noqa: BLE001
        return f"Error reading {path}: {exc}"


def write_file(path: str, content: str) -> str:
    """Write *content* to *path*, creating parent directories as needed."""
    try:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content, encoding="utf-8")
        return f"Successfully wrote {len(content)} characters to {path}"
    except PermissionError:
        return f"Error: permission denied: {path}"
    except Exception as exc:  # noqa: BLE001
        return f"Error writing {path}: {exc}"


def create_file(path: str, content: str = "") -> str:
    """Create a new file at *path*. Returns an error if the file already exists."""
    p = Path(path)
    if p.exists():
        return f"Error: file already exists: {path}"
    return write_file(path, content)


def list_directory(path: str = ".") -> str:
    """Return a formatted directory listing of *path*."""
    try:
        p = Path(path)
        if not p.exists():
            return f"Error: path does not exist: {path}"
        if not p.is_dir():
            return f"Error: not a directory: {path}"
        entries = sorted(p.iterdir(), key=lambda e: (e.is_file(), e.name))
        lines = []
        for entry in entries:
            prefix = "📄" if entry.is_file() else "📁"
            lines.append(f"{prefix} {entry.name}")
        return "\n".join(lines) if lines else "(empty directory)"
    except PermissionError:
        return f"Error: permission denied: {path}"
    except Exception as exc:  # noqa: BLE001
        return f"Error listing {path}: {exc}"


def run_command(command: str, working_dir: str = ".") -> str:
    """Execute *command* in a shell and return combined stdout + stderr."""
    try:
        result = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            cwd=working_dir,
            timeout=60,
        )
        output = ""
        if result.stdout:
            output += result.stdout
        if result.stderr:
            output += result.stderr
        if result.returncode != 0:
            output += f"\n[exit code {result.returncode}]"
        output = output.strip()
        if len(output) > MAX_OUTPUT_CHARS:
            output = output[:MAX_OUTPUT_CHARS] + f"\n... [truncated to {MAX_OUTPUT_CHARS} chars]"
        return output or "(no output)"
    except subprocess.TimeoutExpired:
        return "Error: command timed out after 60 seconds"
    except Exception as exc:  # noqa: BLE001
        return f"Error running command: {exc}"


# ---------------------------------------------------------------------------
# Tool registry – OpenAI function-calling schema
# ---------------------------------------------------------------------------

TOOLS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read the full contents of a file at the given path.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Absolute or relative path to the file to read.",
                    }
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": "Write content to a file, overwriting it if it already exists. "
                           "Parent directories are created automatically.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Path to the file to write.",
                    },
                    "content": {
                        "type": "string",
                        "description": "Full content to write to the file.",
                    },
                },
                "required": ["path", "content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "create_file",
            "description": "Create a new file. Returns an error if the file already exists.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Path to the new file.",
                    },
                    "content": {
                        "type": "string",
                        "description": "Initial content of the new file.",
                    },
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_directory",
            "description": "List the files and subdirectories in a directory.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Path to the directory to list. Defaults to '.'.",
                    }
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_command",
            "description": "Execute a shell command and return its stdout + stderr. "
                           "Commands time out after 60 seconds.",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "description": "Shell command to execute.",
                    },
                    "working_dir": {
                        "type": "string",
                        "description": "Working directory for the command. Defaults to '.'.",
                    },
                },
                "required": ["command"],
            },
        },
    },
]

# Map tool name -> callable
_TOOL_FUNCTIONS: dict[str, Any] = {
    "read_file": read_file,
    "write_file": write_file,
    "create_file": create_file,
    "list_directory": list_directory,
    "run_command": run_command,
}


def execute_tool(name: str, arguments: dict[str, Any]) -> str:
    """Dispatch a tool call by *name* with *arguments* and return the result string."""
    fn = _TOOL_FUNCTIONS.get(name)
    if fn is None:
        return f"Error: unknown tool '{name}'"
    try:
        return fn(**arguments)
    except TypeError as exc:
        return f"Error: invalid arguments for tool '{name}': {exc}"
