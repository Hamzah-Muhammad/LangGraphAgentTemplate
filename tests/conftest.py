"""
Tests must not depend on the developer's own .env.

run.py calls load_dotenv() when it is imported, which copies a local .env (with a real
provider or key in it) into os.environ for the rest of the test session. Before every
test, every setting that .env.example defines is removed, so each test starts from the
template defaults and sets only what it needs. Same result on any machine and in CI.
"""

from pathlib import Path

import pytest

ENV_EXAMPLE = Path(__file__).resolve().parent.parent / ".env.example"
SETTINGS = sorted({
    line.split("=", 1)[0].strip()
    for line in ENV_EXAMPLE.read_text(encoding="utf-8").splitlines()
    if "=" in line and not line.lstrip().startswith("#")
})


@pytest.fixture(autouse=True)
def isolate_from_local_env(monkeypatch):
    for name in SETTINGS:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.delenv("LANGSMITH_TRACING", raising=False)
