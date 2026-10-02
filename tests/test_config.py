from __future__ import annotations

import json
from datetime import timedelta

from veylo.config import load_config


def test_config_precedence(tmp_path, monkeypatch):
    home = tmp_path / "home"
    project = tmp_path / "project"
    (home / ".veylo").mkdir(parents=True)
    (project / ".veylo").mkdir(parents=True)
    (home / ".veylo" / "config.json").write_text(
        json.dumps({"llm": {"provider": "home", "model": "home-model"}}),
        encoding="utf-8",
    )
    (project / ".veylo" / "config.json").write_text(
        json.dumps({"llm": {"provider": "project", "model": "project-model"}}),
        encoding="utf-8",
    )
    (project / ".env").write_text("VEYLO_MODEL=env-file-model\n", encoding="utf-8")
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("VEYLO_PROVIDER", "process")

    config = load_config(
        project_root=project,
        overrides={"llm": {"model": "cli-model"}},
    )

    assert config.llm.provider == "process"
    assert config.llm.model == "cli-model"


def test_provider_specific_api_key(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setenv("VEYLO_PROVIDER", "deepseek")
    monkeypatch.delenv("VEYLO_API_KEY", raising=False)
    monkeypatch.setenv("DEEPSEEK_API_KEY", "deepseek-key")

    config = load_config(project_root=tmp_path)

    assert config.llm.api_key == "deepseek-key"


def test_checkpoint_config_defaults(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path / "home"))

    config = load_config(project_root=tmp_path)

    assert config.checkpoint.max_completed == 10
    assert config.checkpoint.max_resumable == 10
    assert config.checkpoint.ttl_seconds == 604800  # 7 days
    assert config.checkpoint.ttl == timedelta(seconds=604800)


def test_checkpoint_config_from_file(tmp_path, monkeypatch):
    home = tmp_path / "home"
    (home / ".veylo").mkdir(parents=True)
    (home / ".veylo" / "config.json").write_text(
        json.dumps(
            {"checkpoint": {"max_completed": 5, "max_resumable": 20, "ttl_seconds": 604800}}
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("HOME", str(home))

    config = load_config(project_root=tmp_path)

    assert config.checkpoint.max_completed == 5
    assert config.checkpoint.max_resumable == 20
    assert config.checkpoint.ttl_seconds == 604800
    assert config.checkpoint.ttl == timedelta(seconds=604800)
