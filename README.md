# bp-coding – AI Coding Agent

An AI coding agent that uses an LLM (OpenAI-compatible API) with tool-calling
to read, write and execute code on your behalf.

## Features

* **File tools** – `read_file`, `write_file`, `create_file`, `list_directory`
* **Shell tool** – `run_command` (60 s timeout, output truncation)
* **Iterative loop** – the agent keeps calling tools until the task is done or
  `max_iterations` is reached
* **Streaming support** – `CodingAgent.stream()` yields response chunks
* **CLI** – `bp-agent "your task here"`

## Quick start

```bash
pip install -e .
export OPENAI_API_KEY=sk-...
bp-agent "Create a Python script that prints the Fibonacci sequence up to 100"
```

Or pipe a task from stdin:

```bash
echo "Add type hints to src/utils.py" | bp-agent
```

## Configuration

| Environment variable     | Default                        | Description                       |
|--------------------------|--------------------------------|-----------------------------------|
| `OPENAI_API_KEY`         | *(required)*                   | API key                           |
| `OPENAI_BASE_URL`        | `https://api.openai.com/v1`    | API base URL (for local / proxy)  |
| `AGENT_MODEL`            | `gpt-4o`                       | Model name                        |
| `AGENT_MAX_ITERATIONS`   | `20`                           | Max tool-call iterations          |
| `AGENT_MAX_OUTPUT_CHARS` | `8000`                         | Max chars captured from commands  |

All settings can also be overridden via CLI flags – run `bp-agent --help` for
details.

## CLI options

```
usage: bp-agent [-h] [--model MODEL] [--api-key KEY] [--base-url URL]
                [--max-iterations N] [--quiet] [task]

positional arguments:
  task                  Task to perform (reads from stdin if omitted)

options:
  --model MODEL         LLM model (default: gpt-4o)
  --api-key KEY         OpenAI API key
  --base-url URL        API base URL
  --max-iterations N    Max iterations (default: 20)
  --quiet               Suppress tool-call output
```

## Python API

```python
from agent import CodingAgent

agent = CodingAgent(api_key="sk-...", verbose=True)
result = agent.run("Write a unit test for the add() function in math_utils.py")
print(result)
```

## Running tests

```bash
pip install pytest
pytest
```

## Project structure

```
agent/
  __init__.py   – public exports
  agent.py      – CodingAgent class and main loop
  tools.py      – tool implementations + OpenAI schema
  config.py     – configuration (env vars / defaults)
cli.py          – command-line entry point
tests/
  test_agent.py – agent loop tests (mocked LLM)
  test_tools.py – tool implementation tests
requirements.txt
pyproject.toml
```