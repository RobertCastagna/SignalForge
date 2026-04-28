#!/usr/bin/env python3
"""Local setup helper for the AMDC Streamlit app."""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def _run(command: list[str]) -> None:
    print(f"+ {' '.join(command)}", flush=True)
    subprocess.run(command, cwd=ROOT, check=True)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Create the local AMDC environment and install browser assets."
    )
    parser.add_argument(
        "--start",
        action="store_true",
        help="Start the Streamlit app after setup finishes.",
    )
    args = parser.parse_args()

    if shutil.which("uv") is None:
        print(
            "uv is not installed. Install it first:\n"
            "  curl -LsSf https://astral.sh/uv/install.sh | sh\n"
            "Then open a new terminal and run this setup command again.",
            file=sys.stderr,
        )
        return 1

    _run(["uv", "sync", "--python", "3.12"])
    _run(["uv", "run", "playwright", "install", "chromium"])

    if args.start:
        _run(["uv", "run", "streamlit", "run", "streamlit_app.py"])
        return 0

    print(
        "\nSetup complete.\n"
        "Start the app with:\n"
        "  source .venv/bin/activate\n"
        "  uv run streamlit run streamlit_app.py\n"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
