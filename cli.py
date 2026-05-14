#!/usr/bin/env python3
"""Command-line interface for the AI coding agent."""

from __future__ import annotations

import argparse
import sys

from agent import CodingAgent
from agent.config import MAX_ITERATIONS, MODEL, OPENAI_API_KEY


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="bp-agent",
        description="AI Coding Agent – give it a task and watch it work.",
    )
    parser.add_argument(
        "task",
        nargs="?",
        help="Task for the agent to perform. If omitted, reads from stdin.",
    )
    parser.add_argument(
        "--model",
        default=MODEL,
        metavar="MODEL",
        help=f"LLM model to use (default: {MODEL}).",
    )
    parser.add_argument(
        "--api-key",
        default=None,
        metavar="KEY",
        help="OpenAI API key (default: $OPENAI_API_KEY).",
    )
    parser.add_argument(
        "--base-url",
        default=None,
        metavar="URL",
        help="Base URL for the API endpoint.",
    )
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=MAX_ITERATIONS,
        metavar="N",
        help=f"Maximum agent iterations (default: {MAX_ITERATIONS}).",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress tool call output (verbose mode off).",
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    # Resolve task
    if args.task:
        task = args.task
    elif not sys.stdin.isatty():
        task = sys.stdin.read().strip()
    else:
        parser.print_help()
        return 1

    if not task:
        print("Error: no task provided.", file=sys.stderr)
        return 1

    api_key = args.api_key or OPENAI_API_KEY
    if not api_key:
        print(
            "Error: no API key found. Set OPENAI_API_KEY or pass --api-key.",
            file=sys.stderr,
        )
        return 1

    agent = CodingAgent(
        api_key=api_key,
        base_url=args.base_url,
        model=args.model,
        max_iterations=args.max_iterations,
        verbose=not args.quiet,
    )

    result = agent.run(task)
    print(result)
    return 0


if __name__ == "__main__":
    sys.exit(main())
