"""
One-time setup for a fresh clone. Safe to run again.

    python scripts/setup.py

  1. Turns on the secret guard: git config core.hooksPath .githooks
     (git does not keep hook settings in the repo, so every clone must do this once).
  2. Creates .env from .env.example if there is no .env yet. It never overwrites one.

It prints what it did and what it skipped. It changes nothing else.
"""

import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def enable_secret_guard(root: Path = ROOT) -> str:
    if not (root / ".git").exists():
        return "skipped: not a git repository (run `git init`, then this script again)"
    if not (root / ".githooks").is_dir():
        return "skipped: no .githooks folder"
    current = subprocess.run(
        ["git", "config", "--get", "core.hooksPath"], cwd=root, capture_output=True, text=True
    ).stdout.strip()
    if current == ".githooks":
        return "secret guard already on"
    subprocess.run(["git", "config", "core.hooksPath", ".githooks"], cwd=root, check=True)
    return "secret guard turned on (core.hooksPath = .githooks)"


def create_env(root: Path = ROOT) -> str:
    target, template = root / ".env", root / ".env.example"
    if target.exists():
        return ".env already exists, left alone"
    if not template.is_file():
        return "skipped: no .env.example"
    shutil.copyfile(template, target)
    return ".env created from .env.example: open it and pick your model"


def main() -> int:
    for step in (enable_secret_guard, create_env):
        print(step())
    return 0


if __name__ == "__main__":
    sys.exit(main())
