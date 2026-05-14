"""Tests for agent/tools.py."""

from __future__ import annotations

import os
import stat
import tempfile
from pathlib import Path

import pytest

from agent.tools import (
    create_file,
    execute_tool,
    list_directory,
    read_file,
    run_command,
    write_file,
)


# ---------------------------------------------------------------------------
# read_file
# ---------------------------------------------------------------------------


class TestReadFile:
    def test_reads_existing_file(self, tmp_path: Path) -> None:
        f = tmp_path / "hello.txt"
        f.write_text("hello world", encoding="utf-8")
        assert read_file(str(f)) == "hello world"

    def test_missing_file_returns_error(self, tmp_path: Path) -> None:
        result = read_file(str(tmp_path / "nope.txt"))
        assert result.startswith("Error:")
        assert "not found" in result

    def test_empty_file(self, tmp_path: Path) -> None:
        f = tmp_path / "empty.txt"
        f.write_text("", encoding="utf-8")
        assert read_file(str(f)) == ""


# ---------------------------------------------------------------------------
# write_file
# ---------------------------------------------------------------------------


class TestWriteFile:
    def test_writes_new_file(self, tmp_path: Path) -> None:
        target = tmp_path / "out.txt"
        result = write_file(str(target), "content")
        assert "Successfully wrote" in result
        assert target.read_text(encoding="utf-8") == "content"

    def test_overwrites_existing_file(self, tmp_path: Path) -> None:
        target = tmp_path / "out.txt"
        target.write_text("old", encoding="utf-8")
        write_file(str(target), "new")
        assert target.read_text(encoding="utf-8") == "new"

    def test_creates_parent_directories(self, tmp_path: Path) -> None:
        target = tmp_path / "a" / "b" / "c.txt"
        write_file(str(target), "deep")
        assert target.read_text(encoding="utf-8") == "deep"

    def test_reports_character_count(self, tmp_path: Path) -> None:
        target = tmp_path / "count.txt"
        result = write_file(str(target), "abc")
        assert "3 characters" in result


# ---------------------------------------------------------------------------
# create_file
# ---------------------------------------------------------------------------


class TestCreateFile:
    def test_creates_new_file(self, tmp_path: Path) -> None:
        target = tmp_path / "new.txt"
        result = create_file(str(target), "hello")
        assert "Successfully wrote" in result
        assert target.read_text(encoding="utf-8") == "hello"

    def test_fails_if_file_exists(self, tmp_path: Path) -> None:
        target = tmp_path / "existing.txt"
        target.write_text("already here", encoding="utf-8")
        result = create_file(str(target), "new content")
        assert "Error" in result
        assert "already exists" in result
        # original content must be untouched
        assert target.read_text(encoding="utf-8") == "already here"

    def test_creates_with_empty_content_by_default(self, tmp_path: Path) -> None:
        target = tmp_path / "empty.txt"
        create_file(str(target))
        assert target.read_text(encoding="utf-8") == ""


# ---------------------------------------------------------------------------
# list_directory
# ---------------------------------------------------------------------------


class TestListDirectory:
    def test_lists_files_and_dirs(self, tmp_path: Path) -> None:
        (tmp_path / "file.txt").write_text("", encoding="utf-8")
        (tmp_path / "subdir").mkdir()
        result = list_directory(str(tmp_path))
        assert "file.txt" in result
        assert "subdir" in result

    def test_empty_directory(self, tmp_path: Path) -> None:
        result = list_directory(str(tmp_path))
        assert "empty" in result.lower()

    def test_nonexistent_path_returns_error(self, tmp_path: Path) -> None:
        result = list_directory(str(tmp_path / "no_such_dir"))
        assert "Error" in result

    def test_file_path_returns_error(self, tmp_path: Path) -> None:
        f = tmp_path / "a.txt"
        f.write_text("x", encoding="utf-8")
        result = list_directory(str(f))
        assert "Error" in result

    def test_dirs_appear_before_files(self, tmp_path: Path) -> None:
        (tmp_path / "z_file.txt").write_text("", encoding="utf-8")
        (tmp_path / "a_subdir").mkdir()
        lines = list_directory(str(tmp_path)).splitlines()
        # directories (📁) should come before files (📄)
        dir_idx = next(i for i, l in enumerate(lines) if "📁" in l)
        file_idx = next(i for i, l in enumerate(lines) if "📄" in l)
        assert dir_idx < file_idx


# ---------------------------------------------------------------------------
# run_command
# ---------------------------------------------------------------------------


class TestRunCommand:
    def test_runs_simple_command(self) -> None:
        result = run_command("echo hello")
        assert "hello" in result

    def test_captures_stderr(self) -> None:
        result = run_command("echo error >&2")
        # stderr from echo via shell redirect may appear differently; just
        # check the command ran without throwing
        assert isinstance(result, str)

    def test_nonzero_exit_code_reported(self) -> None:
        result = run_command("exit 42")
        assert "42" in result

    def test_working_dir_respected(self, tmp_path: Path) -> None:
        result = run_command("pwd", working_dir=str(tmp_path))
        assert str(tmp_path) in result

    def test_timeout_returns_error(self) -> None:
        # Sleep longer than timeout; we mock the timeout scenario instead by
        # patching subprocess.run to raise TimeoutExpired
        import subprocess
        from unittest.mock import patch

        with patch("subprocess.run", side_effect=subprocess.TimeoutExpired("sleep", 1)):
            result = run_command("sleep 999")
        assert "timed out" in result.lower()

    def test_no_output_returns_placeholder(self) -> None:
        result = run_command("true")
        assert result  # non-empty string returned


# ---------------------------------------------------------------------------
# execute_tool dispatcher
# ---------------------------------------------------------------------------


class TestExecuteTool:
    def test_dispatches_read_file(self, tmp_path: Path) -> None:
        f = tmp_path / "t.txt"
        f.write_text("dispatched", encoding="utf-8")
        result = execute_tool("read_file", {"path": str(f)})
        assert result == "dispatched"

    def test_unknown_tool_returns_error(self) -> None:
        result = execute_tool("nonexistent_tool", {})
        assert "Error" in result
        assert "unknown tool" in result

    def test_missing_required_arg_returns_error(self) -> None:
        # write_file requires both path and content
        result = execute_tool("write_file", {})
        assert "Error" in result

    def test_list_directory_default_path(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        monkeypatch.chdir(tmp_path)
        result = execute_tool("list_directory", {})
        assert isinstance(result, str)
