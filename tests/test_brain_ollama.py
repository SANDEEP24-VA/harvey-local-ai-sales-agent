"""Tests for Ollama local AI provider integration."""

import asyncio
import json
import os
import tempfile
from unittest.mock import AsyncMock, patch

import httpx
import pytest
import pytest_asyncio
from pydantic import ValidationError

from sellbird.brain import Brain
from sellbird.config import BrainConfig, ConfigError, SellbirdConfig, load_brain_config
from sellbird.state import StateManager

BASE_CONFIG = {
    "persona": {"name": "Alex", "company": "Acme", "role": "BDR", "email": "alex@acme.com", "linkedin": "", "tone": "friendly"},
    "product": {"name": "SalesTool", "description": "AI sales tool", "pricing": "$50/mo", "key_benefits": ["Speed"], "objection_responses": {}},
    "icp": {"industries": ["Tech"], "company_size": "10-50", "titles": ["VP Sales"], "geography": ["US"]},
}


@pytest_asyncio.fixture
async def state():
    with tempfile.TemporaryDirectory() as tmpdir:
        sm = StateManager(os.path.join(tmpdir, "test.db"))
        await sm.init_db()
        yield sm


# ── Configuration & Validation ──


def test_ollama_is_default_provider():
    cfg = BrainConfig()
    assert cfg.provider == "ollama"
    assert cfg.ollama_base_url == "http://127.0.0.1:11434"
    assert cfg.ollama_model == "qwen3:4b-instruct-2507-q4_K_M"
    assert cfg.ollama_timeout == 120.0
    assert cfg.ollama_temperature == 0.2
    assert cfg.ollama_context_length == 8192


def test_sellbird_config_defaults_to_ollama():
    config = SellbirdConfig(**BASE_CONFIG)
    assert config.brain.provider == "ollama"
    assert config.brain.ollama_model == "qwen3:4b-instruct-2507-q4_K_M"


def test_load_brain_config_missing_file_defaults_to_ollama(tmp_path):
    path = tmp_path / "sellbird.yaml"
    cfg = load_brain_config(str(path))
    assert cfg.provider == "ollama"


def test_load_brain_config_empty_brain_section_defaults_to_ollama(tmp_path):
    import yaml
    path = tmp_path / "sellbird.yaml"
    path.write_text(yaml.dump({"persona": {"name": "Test"}}))
    cfg = load_brain_config(str(path))
    assert cfg.provider == "ollama"


def test_ollama_endpoint_loopback_validation():
    # Valid loopback addresses
    assert BrainConfig(ollama_base_url="http://127.0.0.1:11434").ollama_base_url == "http://127.0.0.1:11434"
    assert BrainConfig(ollama_base_url="http://localhost:11434").ollama_base_url == "http://localhost:11434"
    assert BrainConfig(ollama_base_url="http://[::1]:11434").ollama_base_url == "http://[::1]:11434"
    assert BrainConfig(ollama_base_url="http://127.0.0.2:11434").ollama_base_url == "http://127.0.0.2:11434"

    # Non-loopback addresses must be rejected
    for invalid_url in (
        "http://192.168.1.100:11434",
        "http://10.0.0.1:11434",
        "http://example.com:11434",
        "https://api.ollama.com",
        "ftp://127.0.0.1:11434",
    ):
        with pytest.raises(ValidationError):
            BrainConfig(ollama_base_url=invalid_url)


def test_ollama_model_name_validation():
    # Valid model names
    assert BrainConfig(ollama_model="qwen3:4b-instruct-2507-q4_K_M").ollama_model == "qwen3:4b-instruct-2507-q4_K_M"
    assert BrainConfig(ollama_model="llama3.2:1b").ollama_model == "llama3.2:1b"
    assert BrainConfig(ollama_model="mistral").ollama_model == "mistral"

    # Invalid model names (command injection or flag attempts)
    for bad in ("-v", "--flag", "model; rm -rf /", "model$(whoami)", ""):
        with pytest.raises(ValidationError):
            BrainConfig(ollama_model=bad)


def test_ollama_numeric_parameters_validation():
    # Invalid timeout
    with pytest.raises(ValidationError):
        BrainConfig(ollama_timeout=0)
    with pytest.raises(ValidationError):
        BrainConfig(ollama_timeout=-10)

    # Invalid temperature
    with pytest.raises(ValidationError):
        BrainConfig(ollama_temperature=-0.1)
    with pytest.raises(ValidationError):
        BrainConfig(ollama_temperature=2.5)

    # Invalid context length
    with pytest.raises(ValidationError):
        BrainConfig(ollama_context_length=0)
    with pytest.raises(ValidationError):
        BrainConfig(ollama_context_length=-100)


# ── Successful Ollama Inference ──


@pytest.mark.asyncio
async def test_successful_ollama_inference(state):
    captured_requests = []

    async def fake_post(url, **kwargs):
        captured_requests.append({"url": url, "json": kwargs.get("json")})
        resp = httpx.Response(
            200,
            json={
                "model": "qwen3:4b-instruct-2507-q4_K_M",
                "message": {"role": "assistant", "content": "I would love to help you scale sales."},
                "done": True,
                "total_duration": 300_000_000,
                "prompt_eval_count": 42,
                "eval_count": 18,
            },
            request=httpx.Request("POST", url),
        )
        return resp

    brain = Brain(state)
    assert brain.provider == "ollama"

    with patch.object(httpx.AsyncClient, "post", side_effect=fake_post):
        reply = await brain.think("Write an intro email", agent="writer", task="draft")

    assert reply == "I would love to help you scale sales."
    assert len(captured_requests) == 1
    req = captured_requests[0]
    assert req["url"] == "http://127.0.0.1:11434/api/chat"
    assert req["json"]["model"] == "qwen3:4b-instruct-2507-q4_K_M"
    assert req["json"]["stream"] is False
    assert req["json"]["options"]["temperature"] == 0.2
    assert req["json"]["options"]["num_ctx"] == 8192
    assert req["json"]["messages"] == [{"role": "user", "content": "Write an intro email"}]
    assert "format" not in req["json"]

    # Verify usage accounting
    events = await state.usage_by_provider(days=1)
    assert any(e["provider"] == "ollama" for e in events)


# ── JSON-mode Inference ──


@pytest.mark.asyncio
async def test_json_mode_inference(state):
    captured_requests = []

    async def fake_post(url, **kwargs):
        captured_requests.append({"url": url, "json": kwargs.get("json")})
        return httpx.Response(
            200,
            json={
                "model": "qwen3:4b-instruct-2507-q4_K_M",
                "message": {"role": "assistant", "content": '{"intent": "interested", "score": 95}'},
                "done": True,
            },
            request=httpx.Request("POST", url),
        )

    brain = Brain(state)

    with patch.object(httpx.AsyncClient, "post", side_effect=fake_post):
        result = await brain.think_json("Classify this reply: Yes let's meet")

    assert result == {"intent": "interested", "score": 95}
    assert len(captured_requests) == 1
    assert captured_requests[0]["json"]["format"] == "json"
    assert brain.last_reply_empty is False


# ── Health Check & Preflight ──


@pytest.mark.asyncio
async def test_ollama_preflight_success(state):
    async def fake_get(url, **kwargs):
        return httpx.Response(
            200,
            json={
                "models": [
                    {"name": "qwen3:4b-instruct-2507-q4_K_M:latest"},
                    {"name": "llama3:latest"},
                ]
            },
            request=httpx.Request("GET", url),
        )

    brain = Brain(state)
    with patch.object(httpx.AsyncClient, "get", side_effect=fake_get):
        err = await brain.preflight()

    assert err == ""
    assert brain.paused_reason == ""


@pytest.mark.asyncio
async def test_ollama_preflight_model_not_installed(state):
    async def fake_get(url, **kwargs):
        return httpx.Response(
            200,
            json={"models": [{"name": "mistral:latest"}]},
            request=httpx.Request("GET", url),
        )

    brain = Brain(state)
    with patch.object(httpx.AsyncClient, "get", side_effect=fake_get):
        err = await brain.preflight()

    assert "not installed" in err
    assert "ollama pull" in err
    assert "qwen3:4b-instruct-2507-q4_K_M" in err
    assert brain.paused_reason != ""


@pytest.mark.asyncio
async def test_ollama_preflight_daemon_unreachable(state):
    async def fake_get(url, **kwargs):
        raise httpx.ConnectError("Connection refused")

    brain = Brain(state)
    with patch.object(httpx.AsyncClient, "get", side_effect=fake_get):
        err = await brain.preflight()

    assert "not reachable" in err
    assert "Ensure Ollama is running" in err
    assert brain.paused_reason != ""


# ── Failure Handling & Zero Cloud Fallback ──


@pytest.mark.asyncio
async def test_no_cloud_fallback_when_ollama_fails(state):
    """When Ollama fails, it MUST NEVER fall back to Claude or Codex."""
    async def fake_post(url, **kwargs):
        raise httpx.ConnectError("Daemon down")

    async def fake_subprocess(*cmd, **kwargs):
        raise AssertionError("CRITICAL: Attempted to run subprocess/Claude when Ollama failed!")

    brain = Brain(state)
    with patch.object(httpx.AsyncClient, "post", side_effect=fake_post), \
         patch("asyncio.create_subprocess_exec", side_effect=fake_subprocess):
        reply = await brain.think("Hello", max_retries=0)

    assert reply == ""
    assert brain.paused_reason != ""


@pytest.mark.asyncio
async def test_ollama_http_404_stops_retries_and_marks_unavailable(state):
    calls = 0

    async def fake_post(url, **kwargs):
        nonlocal calls
        calls += 1
        return httpx.Response(404, text="model 'qwen3:4b-instruct-2507-q4_K_M' not found", request=httpx.Request("POST", url))

    brain = Brain(state)
    with patch.object(httpx.AsyncClient, "post", side_effect=fake_post):
        reply = await brain.think("Hello", max_retries=2)

    assert reply == ""
    assert calls == 1  # 404 does not retry
    assert "not found" in brain.paused_reason


@pytest.mark.asyncio
async def test_ollama_http_500_retries_and_recovers(state):
    calls = 0

    async def fake_post(url, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            return httpx.Response(500, text="Internal Server Error", request=httpx.Request("POST", url))
        return httpx.Response(
            200,
            json={"message": {"content": "Recovered"}},
            request=httpx.Request("POST", url),
        )

    brain = Brain(state)
    with patch.object(httpx.AsyncClient, "post", side_effect=fake_post), \
         patch("asyncio.sleep", AsyncMock()):
        reply = await brain.think("Hello", max_retries=2)

    assert reply == "Recovered"
    assert calls == 2


@pytest.mark.asyncio
async def test_ollama_timeout_handling(state):
    calls = 0

    async def fake_post(url, **kwargs):
        nonlocal calls
        calls += 1
        raise httpx.TimeoutException("Request timed out")

    brain = Brain(state)
    with patch.object(httpx.AsyncClient, "post", side_effect=fake_post), \
         patch("asyncio.sleep", AsyncMock()):
        reply = await brain.think("Hello", max_retries=1)

    assert reply == ""
    assert calls == 2


@pytest.mark.asyncio
async def test_budget_and_quota_always_permits_ollama(state):
    """Local inference has no subscription caps; is_within_budget must always return True."""
    brain = Brain(state)
    assert await brain.is_within_budget(max_daily_calls=0) is True
    assert await brain.is_within_budget(max_percent=0.0) is True
    assert await brain.check_usage() == 0.0


@pytest.mark.asyncio
async def test_proxy_disabled_for_ollama(state):
    """Environment proxies must be disabled (trust_env=False)."""
    client_instances = []

    original_init = httpx.AsyncClient.__init__

    def spy_init(self, *args, **kwargs):
        client_instances.append(kwargs)
        original_init(self, *args, **kwargs)

    async def fake_post(url, **kwargs):
        return httpx.Response(200, json={"message": {"content": "ok"}}, request=httpx.Request("POST", url))

    brain = Brain(state)
    with patch.object(httpx.AsyncClient, "__init__", spy_init), \
         patch.object(httpx.AsyncClient, "post", side_effect=fake_post):
        await brain.think("test", max_retries=0)

    assert len(client_instances) > 0
    assert client_instances[0].get("trust_env") is False
