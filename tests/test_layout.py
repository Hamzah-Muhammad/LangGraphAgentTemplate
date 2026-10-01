"""The layout is part of the template. These tests stop it drifting."""

import json
from pathlib import Path

from agent.prompt.loader import PROMPT_PATH
from agent.prompt.skills import SKILLS_DIR
from agent.tools.mcp import CONFIG_PATH

ROOT = Path(__file__).resolve().parent.parent
BLOCK_FOLDERS = ["model", "tools", "prompt", "context", "memory", "orchestration", "shared"]


def test_config_paths_resolve():
    assert PROMPT_PATH.is_file()
    assert SKILLS_DIR.is_dir()
    assert CONFIG_PATH.is_file()


def test_langgraph_json_points_at_real_graphs():
    """`langgraph dev` failed on a stale path once (studio.py moved in the restructure)."""
    graphs = json.loads((ROOT / "langgraph.json").read_text(encoding="utf-8"))["graphs"]
    for name, spec in graphs.items():
        file_part, _, attr = spec.partition(":")
        target = ROOT / file_part
        assert target.is_file(), f"{name}: {file_part} does not exist"
        source = target.read_text(encoding="utf-8")
        defined = f"def {attr}(" in source or f"{attr} =" in source
        assert defined, f"{name}: {attr} not in {file_part}"


def test_agent_has_exactly_one_folder_per_block_plus_shared():
    folders = sorted(p.name for p in (ROOT / "agent").iterdir()
                     if p.is_dir() and p.name != "__pycache__")
    assert folders == sorted(BLOCK_FOLDERS)


def test_readme_tree_lists_every_file_under_agent():
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    tree = readme.split("## Structure", 1)[1].split("```", 2)[1]
    files = [p for p in (ROOT / "agent").rglob("*") if p.is_file() and "__pycache__" not in p.parts]
    missing = [p.relative_to(ROOT).as_posix() for p in files if p.name not in tree]
    assert not missing, f"add these to the README Structure tree: {missing}"


def test_every_module_names_its_block():
    for f in (ROOT / "agent").rglob("*.py"):
        if f.name != "__init__.py" and len(f.relative_to(ROOT).parts) > 2:
            first = f.read_text(encoding="utf-8").splitlines()[0]
            assert first.startswith(f"# {f.relative_to(ROOT).as_posix()}  |"), f
