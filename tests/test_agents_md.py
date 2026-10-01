"""
AGENTS.md is read by coding agents before they touch the repo. It must stay short and
true: every file it names must exist, and the README must point to it.
"""

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TEXT = (ROOT / "AGENTS.md").read_text(encoding="utf-8")
MAX_LINES = 70  # it is loaded into the agent's context every session
REPO_ROOTS = ("agent/", "tests/", "scripts/", "docs/", "evals/", ".githooks/")


def named_paths() -> list[str]:
    """Every backticked token in AGENTS.md that points into the repo."""
    paths = []
    for token in re.findall(r"`([^`\s]+)`", TEXT):
        token = token.split("::")[0].rstrip(",.)")
        if token.startswith(REPO_ROOTS) or token in {"README.md", ".env"}:
            paths.append(token)
    return paths


def test_it_stays_short():
    assert len(TEXT.splitlines()) <= MAX_LINES


def test_every_file_it_names_exists():
    missing = []
    for path in named_paths():
        if path == ".env" or "<" in path:  # .env is local-only; <path> is a placeholder
            continue
        if not (ROOT / path.rstrip("/")).exists():
            missing.append(path)
    assert not missing, f"AGENTS.md names files that do not exist: {missing}"


def test_it_names_a_test_for_every_rule():
    rules = re.findall(r"^\d\. \*\*", TEXT, flags=re.MULTILINE)
    pinned = re.findall(r"`tests/test_\w+\.py`|`scripts/check_secrets\.py`", TEXT)
    assert len(rules) >= 8 and len(pinned) >= len(rules) - 2, "each rule says what pins it"


def test_claude_md_only_forwards_to_agents_md():
    """CLAUDE.md wins over AGENTS.md when both exist, so it must hold no rules of its own:
    a second copy of the rules would drift. It may only import AGENTS.md."""
    claude = (ROOT / "CLAUDE.md").read_text(encoding="utf-8")
    without_comments = re.sub(r"<!--.*?-->", "", claude, flags=re.DOTALL).strip()
    assert without_comments == "@AGENTS.md"


def test_readme_points_to_it():
    assert "AGENTS.md" in (ROOT / "README.md").read_text(encoding="utf-8")


def test_it_holds_no_secret():
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "check_secrets", ROOT / "scripts" / "check_secrets.py")
    guard = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(guard)
    assert guard.find_secrets(TEXT) == []
