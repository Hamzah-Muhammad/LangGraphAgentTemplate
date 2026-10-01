"""scripts/setup.py: the one-time clone setup. It must be safe to run twice."""

import importlib.util
import subprocess
from pathlib import Path

SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "setup.py"
spec = importlib.util.spec_from_file_location("setup_script", SCRIPT)
setup = importlib.util.module_from_spec(spec)
spec.loader.exec_module(setup)


def test_env_is_created_once_and_never_overwritten(tmp_path):
    (tmp_path / ".env.example").write_text("A=1\n", encoding="utf-8")
    assert "created" in setup.create_env(tmp_path)
    (tmp_path / ".env").write_text("A=secret\n", encoding="utf-8")
    assert "left alone" in setup.create_env(tmp_path)
    assert (tmp_path / ".env").read_text(encoding="utf-8") == "A=secret\n"


def test_secret_guard_turns_on_and_second_run_is_a_no_op(tmp_path):
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    (tmp_path / ".githooks").mkdir()
    assert "turned on" in setup.enable_secret_guard(tmp_path)
    got = subprocess.run(["git", "config", "core.hooksPath"], cwd=tmp_path,
                         capture_output=True, text=True).stdout.strip()
    assert got == ".githooks"
    assert "already on" in setup.enable_secret_guard(tmp_path)


def test_outside_a_git_repo_it_skips_instead_of_failing(tmp_path):
    assert "skipped" in setup.enable_secret_guard(tmp_path)
