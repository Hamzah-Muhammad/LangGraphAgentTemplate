"""
No key may ever reach GitHub. This runs in CI on every push.
Fake keys below are BUILT AT RUNTIME, so this file never contains a key-shaped string.
"""

import importlib.util
import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
GUARD_PATH = ROOT / "scripts" / "check_secrets.py"
spec = importlib.util.spec_from_file_location("check_secrets", GUARD_PATH)
guard = importlib.util.module_from_spec(spec)
spec.loader.exec_module(guard)

FILLER = "a1B2c3D4e5F6g7H8i9J0" * 3


def fake(prefix: str) -> str:
    return prefix + FILLER


def test_no_secret_in_any_tracked_file():
    problems = guard.scan_tracked()
    assert not problems, "\n".join(problems)


def test_env_file_is_ignored_and_not_tracked():
    ignored = subprocess.run(["git", "check-ignore", "-q", ".env"], cwd=ROOT, check=False)
    assert ignored.returncode == 0, ".env must be in .gitignore"
    tracked = subprocess.run(["git", "ls-files"], cwd=ROOT, capture_output=True, text=True,
                             check=False).stdout.splitlines()
    assert ".env" not in tracked
    assert not [p for p in tracked if guard.is_forbidden_file(p)]


def test_env_example_ships_with_every_secret_blank():
    secret_name = re.compile(r"^([A-Z0-9_]*(?:API_KEY|TOKEN|SECRET|PASSWORD))=(.*)$")
    for line in (ROOT / ".env.example").read_text(encoding="utf-8").splitlines():
        match = secret_name.match(line.strip())
        if match and not match.group(1).endswith("_TOKENS"):  # CONTEXT_WINDOW_TOKENS is a number
            assert match.group(2) == "", f"{match.group(1)} must be empty in .env.example"


def test_guard_catches_every_key_family():
    samples = {
        "Anthropic API key": fake("sk-ant-api03-"),
        "OpenAI-style API key": fake("sk-proj-"),
        "Groq API key": fake("gsk_"),
        "NVIDIA API key": fake("nvapi-"),
        "LangSmith API key": fake("lsv2_pt_"),
        "GitHub token": fake("ghp_"),
        "AWS access key": "AKIA" + "ABCDEFGHIJKLMNOP",
        "private key": "-----BEGIN " + "RSA PRIVATE KEY-----",
        "secret assigned to a setting": "OPENAI_API_KEY" + "=" + "whatever-real-value-123",
        "credential in a URL": "https://bob:" + "hunter2hunter2" + "@api.example.org/v1",
    }
    for kind, text in samples.items():
        hits = guard.find_secrets(f"line one\nvalue {text}\n")
        assert hits and hits[0][0] == 2, f"guard missed: {kind}"


def test_guard_never_reports_the_secret_itself():
    secret = fake("sk-ant-api03-")
    assert all(secret not in str(hit) for hit in guard.find_secrets(secret))


def test_guard_allows_normal_template_text():
    harmless = "\n".join([
        "OPENAI_API_KEY=",
        "api_key = os.getenv('OPENAI_API_KEY')",
        "| `OPENAI_API_KEY`, `OPENAI_BASE_URL` | Primary model |",
        'monkeypatch.setenv("OPENAI_API_KEY", "k")',
        "CONTEXT_WINDOW_TOKENS=32000",
        "OPENAI_BASE_URL=https://api.groq.com/openai/v1",
    ])
    assert guard.find_secrets(harmless) == []


def test_env_and_key_files_are_refused_by_name():
    for path in (".env", ".env.local", "config/.env.production", "id.pem", "memory.db"):
        assert guard.is_forbidden_file(path), path
    assert not guard.is_forbidden_file(".env.example")
