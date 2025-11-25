#!/usr/bin/env python3
"""test.py

Quick local tester for Ollama model via the `ollama` CLI.

This script:
- Verifies that the `ollama` CLI is available in PATH.
- Optionally checks whether the requested model appears in `ollama list` output.
- Runs a sample prompt against the model using `ollama run` (tries the common invocation styles).

Usage:
  python test.py --model llama3.2:1b --prompt "Say hello"

Notes:
- This script uses the `ollama` executable (local Ollama install). It does not contact cloud APIs.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from typing import Optional, Tuple


def run_cmd(cmd: list[str], timeout: int = 60) -> Tuple[int, str, str]:
    """Run a command, return (returncode, stdout, stderr)."""
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return proc.returncode, proc.stdout.strip(), proc.stderr.strip()
    except subprocess.TimeoutExpired as e:
        return 124, "", f"timeout after {timeout}s"
    except Exception as e:
        return 1, "", str(e)


def check_ollama_cli() -> bool:
    """Return True if 'ollama' is found in PATH."""
    path = shutil.which("ollama")
    if path:
        print(f"Found ollama CLI at: {path}")
        return True
    print("'ollama' CLI not found in PATH. Install Ollama and ensure 'ollama' is on your PATH.")
    return False


def model_in_list(model: str) -> Optional[str]:
    """Check `ollama list` for the model string. Return stdout if present, else None."""
    rc, out, err = run_cmd(["ollama", "list"])  # list models
    if rc != 0:
        print("Could not run 'ollama list'. stderr:", err)
        return None
    if model in out:
        return out
    return None


def try_run_model(model: str, prompt: str) -> Tuple[bool, str]:
    """Attempt to run the model with two common invocation styles and return (success, output).

    Tries:
      1) ollama run <model> "<prompt>"
      2) ollama run <model> --prompt "<prompt>"
    """
    # style 1: positional prompt
    cmd1 = ["ollama", "run", model, prompt]
    rc1, out1, err1 = run_cmd(cmd1)
    if rc1 == 0 and out1:
        return True, out1

    # style 2: explicit --prompt
    cmd2 = ["ollama", "run", model, "--prompt", prompt]
    rc2, out2, err2 = run_cmd(cmd2)
    if rc2 == 0 and out2:
        return True, out2

    # if neither worked, return combined diagnostics
    diag = "--- Attempt 1 ---\n"
    diag += f"cmd: {' '.join(cmd1)}\nexit: {rc1}\nstdout:\n{out1}\nstderr:\n{err1}\n"
    diag += "\n--- Attempt 2 ---\n"
    diag += f"cmd: {' '.join(cmd2)}\nexit: {rc2}\nstdout:\n{out2}\nstderr:\n{err2}\n"
    return False, diag


def main() -> int:
    parser = argparse.ArgumentParser(description="Quick test for Ollama model via `ollama` CLI")
    parser.add_argument("--model", default="llama3.2:1b", help="Model identifier to test (default: llama3.2:1b)")
    parser.add_argument("--prompt", default="Hello from test script! Please respond concisely.", help="Prompt to send to the model")
    parser.add_argument("--skip-list-check", action="store_true", help="Skip running `ollama list` to check model availability")
    args = parser.parse_args()

    if not check_ollama_cli():
        print("Install instructions: https://ollama.com (or use your platform's package manager).")
        return 2

    if not args.skip_list_check:
        print("Checking installed/pulled models with 'ollama list'...")
        out = model_in_list(args.model)
        if not out:
            print(f"Model '{args.model}' not found in 'ollama list' output. You can pull it with:\n  ollama pull {args.model}\nOr run the script anyway to allow ollama to auto-download if supported.")
        else:
            print(f"Model appears in list. (snippet)\n{out.splitlines()[:20]}")

    print(f"Running model '{args.model}' with prompt: {args.prompt!r}")
    ok, result = try_run_model(args.model, args.prompt)
    if ok:
        print("\n=== MODEL OUTPUT ===\n")
        print(result)
        return 0

    print("\nModel run failed. Diagnostics below:\n")
    print(result)
    return 3


if __name__ == "__main__":
    raise SystemExit(main())


