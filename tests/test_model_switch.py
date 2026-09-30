"""Switching between an API-style LLM and the Claude login must be obvious and safe."""

import importlib.util

import pytest

import run as cli
from agent.model import model as model_module
from agent.model.model import build_model, describe_models

OPENAI_VARS = ("OPENAI_API_KEY", "OPENAI_BASE_URL", "OPENAI_MODEL_NAME")
ROLE_VARS = ("MODEL_PROVIDER", "CLAUDE_MODEL", "FALLBACK_PROVIDER", "FALLBACK_MODEL_NAME",
             "FALLBACK_API_KEY", "FALLBACK_BASE_URL", "GRADER_PROVIDER", "GRADER_MODEL_NAME",
             "GRADER_API_KEY", "GRADER_BASE_URL")


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    for name in OPENAI_VARS + ROLE_VARS:
        monkeypatch.delenv(name, raising=False)


def test_banner_names_the_api_style_model_and_its_host(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "k")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://api.groq.com/openai/v1")
    monkeypatch.setenv("OPENAI_MODEL_NAME", "llama-3.3-70b-versatile")
    assert describe_models() == [
        "[model] primary = openai-compatible: llama-3.3-70b-versatile @ api.groq.com"]


def test_banner_names_the_claude_login_and_every_configured_role(monkeypatch):
    monkeypatch.setenv("MODEL_PROVIDER", "claude-code")
    monkeypatch.setenv("CLAUDE_MODEL", "haiku")
    monkeypatch.setenv("GRADER_PROVIDER", "claude-code")
    monkeypatch.setenv("GRADER_MODEL_NAME", "opus")
    assert describe_models() == [
        "[model] primary = claude-code: haiku (Claude login, no API key)",
        "[model] grader = claude-code: opus (Claude login, no API key)",
    ]


def test_banner_says_so_when_nothing_is_configured():
    assert describe_models() == ["[model] primary = NOT CONFIGURED"]


def test_one_line_switches_the_provider(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "k")
    monkeypatch.setenv("OPENAI_BASE_URL", "http://localhost:1/v1")
    monkeypatch.setenv("OPENAI_MODEL_NAME", "m")
    assert type(build_model()).__name__ == "ChatOpenAI"
    monkeypatch.setenv("MODEL_PROVIDER", "claude-code")  # the one line
    assert type(build_model()).__name__ == "ChatClaudeCode"
    monkeypatch.setenv("MODEL_PROVIDER", "openai")  # and back
    assert type(build_model()).__name__ == "ChatOpenAI"


def test_flags_switch_for_one_run(monkeypatch):
    cli.apply_model_flags("claude-code", "haiku")
    assert describe_models() == ["[model] primary = claude-code: haiku (Claude login, no API key)"]
    monkeypatch.setenv("OPENAI_BASE_URL", "https://api.groq.com/openai/v1")
    cli.apply_model_flags("openai", "llama-x")
    assert describe_models() == ["[model] primary = openai-compatible: llama-x @ api.groq.com"]


def test_missing_api_settings_point_to_the_claude_option():
    with pytest.raises(RuntimeError, match="MODEL_PROVIDER=claude-code"):
        build_model()


def test_missing_claude_sdk_says_how_to_install_it(monkeypatch):
    monkeypatch.setenv("MODEL_PROVIDER", "claude-code")
    real_find_spec = importlib.util.find_spec
    monkeypatch.setattr(
        model_module.importlib.util, "find_spec",
        lambda name, *a, **k: None if name == "claude_agent_sdk" else real_find_spec(name),
    )
    with pytest.raises(RuntimeError, match=r'pip install -e "\.\[claude\]"'):
        build_model()


def test_banner_never_prints_credentials_from_the_model_url(monkeypatch):
    """Was: the banner printed the URL's netloc, which includes user:password@."""
    monkeypatch.setenv("OPENAI_API_KEY", "sk-SECRETKEY")
    monkeypatch.setenv("OPENAI_MODEL_NAME", "m")
    monkeypatch.setenv(
        "OPENAI_BASE_URL", "https://alice:s3cret@proxy.example.com:8443/v1?token=TOPSECRET"
    )
    banner = " ".join(describe_models())
    assert banner == "[model] primary = openai-compatible: m @ proxy.example.com:8443"
    for secret in ("alice", "s3cret", "TOPSECRET", "sk-SECRETKEY"):
        assert secret not in banner


def test_banner_survives_a_malformed_model_url(monkeypatch):
    monkeypatch.setenv("OPENAI_MODEL_NAME", "m")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://host:notaport/v1")
    assert describe_models() == ["[model] primary = openai-compatible: m @ unreadable base URL"]
