"""Brain provider selection and zero-cloud hardening tests.

Validates that:
- Ollama is the default and only supported local LLM provider.
- Claude and Codex configurations are strictly rejected with clear validation errors.
- Unknown providers are rejected.
- Normal production inference exclusively uses Ollama.
- Ollama failures never trigger cloud LLM fallback.
- Setup wizard and trainer configure and persist Ollama.
- Usage tracking reflects Ollama zero-cost operations.
"""

import asyncio
import os
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock

import httpx
import pytest
import pytest_asyncio
import yaml
from fastapi.testclient import TestClient
from pydantic import ValidationError

import sellbird.brain as brain_mod
import sellbird.dashboard as dash
from sellbird.brain import Brain
from sellbird.config import BrainConfig, ConfigError, SellbirdConfig, load_brain_config
from sellbird.state import StateManager

BASE = {
    "persona": {"name": "H", "company": "A", "role": "B", "email": "e", "linkedin": "", "tone": "t"},
    "product": {"name": "P", "description": "D", "pricing": "$1", "key_benefits": [],
                "objection_responses": {}},
    "icp": {"industries": [], "company_size": "", "titles": [], "geography": []},
}


@pytest_asyncio.fixture
async def state():
    with tempfile.TemporaryDirectory() as tmpdir:
        sm = StateManager(os.path.join(tmpdir, "test.db"))
        await sm.init_db()
        yield sm


# ── Provider Configuration & Zero-Cloud Rejection ──


def test_ollama_is_the_default():
    assert SellbirdConfig(**BASE).brain.provider == "ollama"
    assert BrainConfig().provider == "ollama"
    assert BrainConfig().model == ""
    assert BrainConfig().ollama_base_url == "http://127.0.0.1:11434"
    assert BrainConfig().ollama_model == "qwen3:4b-instruct-2507-q4_K_M"


def test_claude_configuration_rejected():
    with pytest.raises(ValidationError) as exc:
        BrainConfig(provider="claude")
    err = str(exc.value)
    assert "disabled" in err
    assert "Only local Ollama inference is supported" in err


def test_codex_configuration_rejected():
    with pytest.raises(ValidationError) as exc:
        BrainConfig(provider="codex")
    err = str(exc.value)
    assert "disabled" in err
    assert "Only local Ollama inference is supported" in err


def test_unknown_providers_rejected():
    for bad in ("openai", "gemini", "anthropic", "azure"):
        with pytest.raises(ValidationError) as exc:
            BrainConfig(provider=bad)
        assert "not a supported brain provider" in str(exc.value)


def test_brain_config_validation():
    cfg = BrainConfig(provider=" ollama ", model="qwen3:4b-instruct-2507-q4_K_M")
    assert (cfg.provider, cfg.model) == ("ollama", "qwen3:4b-instruct-2507-q4_K_M")
    for bad in (
        {"provider": "claude"},
        {"provider": "codex"},
        {"provider": "openai"},
        {"model": "--dangerously-skip-permissions"},
        {"model": "gpt 5; rm -rf /"},
    ):
        with pytest.raises(ValidationError):
            BrainConfig(**bad)


def test_load_brain_config(tmp_path):
    path = tmp_path / "sellbird.yaml"
    assert load_brain_config(str(path)) == BrainConfig()  # missing file defaults to Ollama
    path.write_text(yaml.dump({"persona": "placeholder"}))
    assert load_brain_config(str(path)).provider == "ollama"  # no brain section defaults to Ollama
    path.write_text(yaml.dump({"brain": {"provider": "ollama", "model": "qwen3:4b-instruct-2507-q4_K_M"}}))
    assert load_brain_config(str(path)) == BrainConfig(provider="ollama", model="qwen3:4b-instruct-2507-q4_K_M")

    for disabled in ("claude", "codex", "gemini", "openai"):
        path.write_text(yaml.dump({"brain": {"provider": disabled}}))
        with pytest.raises(ConfigError):  # Cloud and unsupported providers rejected
            load_brain_config(str(path))


# ── Production Ollama Execution & Zero Cloud Fallback ──


@pytest.mark.asyncio
async def test_brain_production_exclusively_uses_ollama(state, monkeypatch):
    """Normal production inference requests exclusively use Ollama."""
    mock_resp = {
        "message": {"content": "Hello from local Ollama model"},
        "prompt_eval_count": 12,
        "eval_count": 8,
    }

    async def fake_post(client, url, *args, **kwargs):
        assert "11434" in str(url)
        return httpx.Response(200, json=mock_resp, request=httpx.Request("POST", url))

    monkeypatch.setattr(brain_mod.httpx.AsyncClient, "post", fake_post)

    # Verify that external subprocesses are never touched
    async def forbidden_exec(*args, **kwargs):
        raise AssertionError("create_subprocess_exec must never be called during normal Ollama inference")

    monkeypatch.setattr(brain_mod.asyncio, "create_subprocess_exec", forbidden_exec)

    brain = Brain(state, BrainConfig())
    resp = await brain.think("Hello", agent="scout", task="evaluate")
    assert resp == "Hello from local Ollama model"

    # Verify usage recorded as Ollama with zero cost
    usage_events = await state.usage_by_provider(days=1)
    assert len(usage_events) == 1
    assert usage_events[0]["provider"] == "ollama"
    assert usage_events[0]["cost_usd"] == 0.0


@pytest.mark.asyncio
async def test_ollama_failure_never_triggers_external_inference(state, monkeypatch):
    """When Ollama fails, inference fails closed and never calls Claude or Codex."""
    async def fake_failing_post(client, url, *args, **kwargs):
        raise httpx.ConnectError("Connection refused by Ollama daemon")

    monkeypatch.setattr(brain_mod.httpx.AsyncClient, "post", fake_failing_post)

    # Disallow any external CLI or cloud subprocess
    async def forbidden_exec(*args, **kwargs):
        raise AssertionError("External cloud subprocess must NEVER be triggered as a fallback")

    monkeypatch.setattr(brain_mod.asyncio, "create_subprocess_exec", forbidden_exec)

    brain = Brain(state, BrainConfig(provider="ollama"))
    assert await brain.think("sensitive prospect reply", max_retries=1) == ""
    assert await brain.think_json("classify sentiment") is None
    assert brain.last_reply_empty is True


@pytest.mark.asyncio
async def test_unsupported_provider_fails_closed(state):
    """Unknown or unsupported providers fail closed without calling anything."""
    fake_config = type("C", (), {
        "provider": "unsupported_cloud_ai",
        "model": "gpt-99",
        "ollama_base_url": "http://127.0.0.1:11434",
    })()
    brain = Brain(state, brain_config=fake_config)
    assert "Unsupported brain provider" in brain.unavailable
    assert await brain.think("test") == ""
    assert await brain.think_json("test") is None
    preflight_err = await brain.preflight()
    assert "Unsupported brain provider" in preflight_err


@pytest.mark.asyncio
async def test_zero_cost_usage_accounting(state, monkeypatch):
    """Ollama calls are tracked with 0 cost and no subscription budget limits."""
    brain = Brain(state, BrainConfig())
    assert await brain.check_usage() == 0.0
    assert await brain.is_within_budget(max_daily_calls=100) is True


# ── Training, Setup and CLI Hardening ──


def test_trainer_uses_and_writes_back_the_selection(tmp_path, monkeypatch):
    import sellbird.trainer as trainer_mod

    monkeypatch.setattr(trainer_mod, "load_brain_config",
                        lambda: BrainConfig(provider="ollama", model="qwen3:4b-instruct-2507-q4_K_M"))
    trainer = trainer_mod.Trainer()
    assert trainer.brain.provider == "ollama" and trainer.brain.model == "qwen3:4b-instruct-2507-q4_K_M"
    built = trainer._build_config({"product_name": "W"}, {}, {}, "acme.com")
    assert built["brain"] == {"provider": "ollama", "model": "qwen3:4b-instruct-2507-q4_K_M"}
    assert SellbirdConfig(**built).brain == BrainConfig(provider="ollama", model="qwen3:4b-instruct-2507-q4_K_M")

    explicit = trainer_mod.Trainer(brain_config=BrainConfig(provider="ollama"))
    assert explicit.brain.provider == "ollama"
    assert explicit._build_config({}, {}, {}, "acme.com")["brain"] == {"provider": "ollama"}


def test_cli_train_flags_select_the_provider(monkeypatch):
    import sellbird.cli as cli
    import sellbird.trainer as trainer_mod
    import sellbird.config as config_mod

    seen = {}

    class FakeTrainer:
        def __init__(self, brain_config=None):
            seen["config"] = brain_config

        async def train(self, url, max_pages=100):
            seen["url"] = url

    monkeypatch.setattr(trainer_mod, "Trainer", FakeTrainer)
    monkeypatch.setattr(config_mod, "load_brain_config",
                        lambda: BrainConfig(provider="ollama", model="qwen3:4b-instruct-2507-q4_K_M"))

    args = type("A", (), {"url": "acme.com", "max_pages": 5, "provider": "ollama",
                          "model": "qwen3:4b-instruct-2507-q4_K_M", "reasoning_effort": None})()
    cli.cmd_train(args)
    assert seen["config"] == BrainConfig(provider="ollama", model="qwen3:4b-instruct-2507-q4_K_M")


def _answering(answers, prompts):
    answers = iter(answers)

    def ask(prompt, default="", required=True, secret=False):
        prompts.append((prompt, default))
        answer = next(answers)
        return answer or default

    return ask


@pytest.mark.parametrize("answers, expected", [
    (["ollama", ""], BrainConfig(provider="ollama", model="qwen3:4b-instruct-2507-q4_K_M")),
    (["ollama", "custom-model:7b"], BrainConfig(provider="ollama", model="custom-model:7b")),
])
def test_setup_model_prompt(monkeypatch, answers, expected):
    import sellbird.setup as setup_mod

    prompts = []
    monkeypatch.setattr(setup_mod, "_ask", _answering(answers, prompts))
    assert setup_mod._ask_brain_config() == expected


def test_setup_rejects_cloud_providers(monkeypatch):
    import sellbird.setup as setup_mod

    # Setup refuses 'claude' and 'codex', continuing loop until 'ollama' is chosen
    answers = iter(["claude", "codex", "ollama", ""])
    monkeypatch.setattr(setup_mod, "_ask", lambda *a, **k: next(answers))
    cfg = setup_mod._ask_brain_config()
    assert cfg.provider == "ollama"


def test_setup_writes_the_selection(tmp_path, monkeypatch):
    import sellbird.setup as setup_mod

    answers = iter(["ollama", "qwen3:4b-instruct-2507-q4_K_M"])
    monkeypatch.setattr(setup_mod, "_ask", lambda *a, **k: next(answers))
    wizard = setup_mod.SetupWizard()
    wizard.brain_config = setup_mod._ask_brain_config()
    assert wizard.brain_config == BrainConfig(provider="ollama", model="qwen3:4b-instruct-2507-q4_K_M")

    config_file = tmp_path / "sellbird.yaml"
    monkeypatch.setattr(setup_mod, "CONFIG_FILE", config_file)
    wizard.config = dict(BASE)
    wizard._write_config()
    assert yaml.safe_load(config_file.read_text())["brain"] == {
        "provider": "ollama", "model": "qwen3:4b-instruct-2507-q4_K_M"
    }


# ── Dashboard & State Compatibility ──


@pytest.fixture
def client(monkeypatch, tmp_path):
    """An authorised same-origin client (the dashboard's own UI)."""
    db = tmp_path / "sellbird.db"
    asyncio.run(StateManager(str(db)).init_db())
    monkeypatch.setattr(dash, "DB_PATH", db)
    monkeypatch.setattr(dash, "_quota_clients", {})
    dash.token_store.configure(tmp_path / "data" / "dashboard-token")
    token = dash.token_store.get()
    with TestClient(dash.app, base_url="http://127.0.0.1:5555",
                    headers={"Authorization": f"Bearer {token}"}) as c:
        c.db_path = db
        yield c


def test_dashboard_ollama_provider_info(client, monkeypatch):
    monkeypatch.setattr(dash, "_active_brain", lambda: BrainConfig(provider="ollama"))
    data = client.get("/api/usage").json()
    assert data["provider"] == "ollama"
    assert data["quota"] is None

    stats = client.get("/api/stats").json()
    assert stats["brain_provider"] == "ollama"


@pytest.mark.asyncio
async def test_historical_usage_rows_preserved(tmp_path):
    """Past database rows from migrations v11 can still be queried safely."""
    from datetime import date
    import aiosqlite
    from sellbird.state import MIGRATIONS

    db_path = str(tmp_path / "old.db")
    async with aiosqlite.connect(db_path) as db:
        for script in MIGRATIONS[:10]:
            await db.executescript(script)
        await db.execute("PRAGMA user_version = 10")
        await db.execute("INSERT INTO usage_events (id, model, output_tokens) VALUES ('a', 'm', 5)")
        await db.execute("INSERT INTO usage_log (id, date, claude_calls) VALUES ('u', ?, 7)",
                         (date.today().isoformat(),))
        await db.commit()

    sm = StateManager(db_path)
    await sm.init_db()
    rows = await sm.usage_by_provider(days=30)
    assert [(r["provider"], r["output_tokens"]) for r in rows] == [("claude", 5)]
    assert await sm.get_usage_today("claude") == 7
    with pytest.raises(ValueError):
        await sm.record_usage_event(provider="gemini")
