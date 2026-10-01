from __future__ import annotations

import json
import os
from contextlib import suppress
from copy import deepcopy
from dataclasses import asdict, dataclass, field
from datetime import timedelta
from pathlib import Path
from typing import Any


def _home() -> Path:
    return Path.home()


@dataclass(slots=True)
class LlmConfig:
    provider: str = "deepseek"
    model: str = "deepseek-v4-flash"
    api_key: str = ""
    base_url: str | None = None
    context_window: int | None = None
    # Optional per-million-token overrides, keyed by currency then
    # input_cache_hit/input_cache_miss/output. Provider prices can change.
    prices: dict[str, dict[str, float]] = field(default_factory=dict)
    max_tokens: int = 8192
    temperature: float = 0.7
    timeout: float = 120.0


@dataclass(slots=True)
class ToolsConfig:
    enabled: list[str] = field(default_factory=list)
    disabled: list[str] = field(default_factory=list)
    timeout: float = 60.0
    batch_timeout: float = 90.0
    max_concurrent_read: int = 4


@dataclass(slots=True)
class McpConfig:
    servers: list[dict[str, Any]] = field(default_factory=list)
    auto_start: bool = True


@dataclass(slots=True)
class MemoryConfig:
    max_conversation_history: int = 100
    long_term_enabled: bool = True
    long_term_db_path: str = "~/.veylo/memory.db"
    max_long_term_entries: int = 1_000
    max_memory_chars: int = 8_000
    recall_limit: int = 6
    recall_min_score: float = 0.05
    token_budget_mode: str = "balanced"
    compression_threshold: float = 0.8
    compression_target: float = 0.55
    compression_reserve_tokens: int = 1_024
    min_recent_messages: int = 6
    summary_max_chars: int = 6_000


@dataclass(slots=True)
class PolicyConfig:
    hitl_mode: str = "auto"
    path_guard_enabled: bool = True
    command_guard_enabled: bool = True
    command_blacklist: list[str] = field(
        default_factory=lambda: [
            "sudo",
            "rm -rf /",
            "rm -rf ~",
            "mkfs",
            "dd if=/dev/zero",
            ":(){:|:&};:",
            "chmod -R 777 /",
            "curl | sh",
            "curl|sh",
            "shutdown",
            "reboot",
        ]
    )
    audit_log_path: str = "~/.veylo/audit.jsonl"


@dataclass(slots=True)
class PromptConfig:
    personality: str = "default"
    agent_mode: str = "react"
    custom_prompt_paths: list[str] = field(default_factory=list)


@dataclass(slots=True)
class FeatureConfig:
    mcp: bool = True
    skill: bool = True
    memory: bool = True
    audit_log: bool = True
    context_compression: bool = True
    code_index: bool = True


@dataclass(slots=True)
class CheckpointConfig:
    """Retention policy for pruning execution checkpoints.

    ``max_completed`` and ``max_resumable`` cap how many finished / unfinished
    runs are kept (oldest dropped first). ``ttl_seconds`` optionally adds a
    time-based cleanup for unfinished runs; ``None`` disables it.
    """

    max_completed: int = 10
    max_resumable: int = 10
    ttl_seconds: int = 7 * 24 * 60 * 60  # 7 days

    @property
    def ttl(self) -> timedelta:
        """The TTL as a ``timedelta``, or ``None`` when time cleanup is off."""
        return timedelta(seconds=self.ttl_seconds)


@dataclass(slots=True)
class VeyloConfig:
    llm: LlmConfig = field(default_factory=LlmConfig)
    render_mode: str = "inline"
    tools: ToolsConfig = field(default_factory=ToolsConfig)
    mcp: McpConfig = field(default_factory=McpConfig)
    memory: MemoryConfig = field(default_factory=MemoryConfig)
    policy: PolicyConfig = field(default_factory=PolicyConfig)
    prompt: PromptConfig = field(default_factory=PromptConfig)
    features: FeatureConfig = field(default_factory=FeatureConfig)
    checkpoint: CheckpointConfig = field(default_factory=CheckpointConfig)


def load_config(
    project_root: str | Path | None = None,
    overrides: dict[str, Any] | None = None,
    env: dict[str, str | None] | None = None,
) -> VeyloConfig:
    env_map = env if env is not None else os.environ
    data = _config_to_dict(VeyloConfig())

    user_config = _read_json(_home() / ".veylo" / "config.json")
    if user_config:
        data = _deep_merge(data, user_config)

    root = Path(project_root).resolve() if project_root else None
    if root:
        project_config = _read_json(root / ".veylo" / "config.json")
        if project_config:
            data = _deep_merge(data, project_config)
        project_env = _read_env(root / ".env")
        if project_env:
            data = _apply_env(data, project_env)

    if overrides:
        data = _deep_merge(data, overrides)

    data = _apply_env(data, env_map)
    config = _dict_to_config(data)
    config.memory.long_term_db_path = _expand_home(config.memory.long_term_db_path)
    config.policy.audit_log_path = _expand_home(config.policy.audit_log_path)
    return config


def get_config_paths(project_root: str | Path | None = None) -> list[Path]:
    paths = [_home() / ".veylo" / "config.json"]
    if project_root:
        paths.append(Path(project_root).resolve() / ".veylo" / "config.json")
    return paths


def config_to_public_dict(config: VeyloConfig) -> dict[str, Any]:
    data = _config_to_dict(config)
    if data.get("llm", {}).get("api_key"):
        data["llm"]["api_key"] = "***"
    return data


def _read_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return raw if isinstance(raw, dict) else None


def _read_env(path: Path) -> dict[str, str]:
    result: dict[str, str] = {}
    if not path.exists():
        return result
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return result
    for raw_line in lines:
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if (value.startswith('"') and value.endswith('"')) or (
            value.startswith("'") and value.endswith("'")
        ):
            value = value[1:-1]
        result[key] = value
    return result


def _apply_env(data: dict[str, Any], env: dict[str, str | None]) -> dict[str, Any]:
    result = deepcopy(data)
    llm = result.setdefault("llm", {})
    features = result.setdefault("features", {})
    policy = result.setdefault("policy", {})

    mappings: list[tuple[str, str, Any]] = [
        ("VEYLO_API_KEY", "api_key", str),
        ("VEYLO_PROVIDER", "provider", str),
        ("VEYLO_MODEL", "model", str),
        ("VEYLO_BASE_URL", "base_url", str),
        ("VEYLO_CONTEXT_WINDOW", "context_window", int),
        ("VEYLO_MAX_TOKENS", "max_tokens", int),
        ("VEYLO_TEMPERATURE", "temperature", float),
    ]
    for env_key, config_key, caster in mappings:
        raw = env.get(env_key)
        if raw not in (None, ""):
            with suppress(TypeError, ValueError):
                llm[config_key] = caster(raw)

    provider = str(llm.get("provider") or "").lower()
    if not llm.get("api_key"):
        provider_key_map = {
            "deepseek": ("DEEPSEEK_API_KEY",),
            "glm": ("ZAI_API_KEY", "GLM_API_KEY"),
            "zhipu": ("ZAI_API_KEY", "GLM_API_KEY"),
            "qwen": ("DASHSCOPE_API_KEY",),
            "kimi": ("KIMI_API_KEY",),
            "moonshot": ("KIMI_API_KEY",),
            "freellmapi": ("FREELLMAPI_API_KEY",),
            "xfyun": ("XFYUN_API_KEY",),
            "agnes": ("AGNES_API_KEY",),
        }
        for provider_key in provider_key_map.get(provider, ()):
            if env.get(provider_key):
                llm["api_key"] = env[provider_key]
                break

    provider_model_key = f"{provider.upper()}_MODEL" if provider else ""
    provider_base_url_key = f"{provider.upper()}_BASE_URL" if provider else ""
    if provider_model_key and env.get(provider_model_key):
        llm["model"] = env[provider_model_key]
    if provider_base_url_key and env.get(provider_base_url_key):
        llm["base_url"] = env[provider_base_url_key]

    render_mode = env.get("VEYLO_RENDER_MODE") or env.get("VEYLO_RENDERER")
    if render_mode in {"plain", "inline"}:
        result["render_mode"] = render_mode

    if env.get("VEYLO_TUI") == "true":
        result["render_mode"] = "inline"

    for env_key, feature_key in [
        ("VEYLO_MCP", "mcp"),
        ("VEYLO_SKILL", "skill"),
        ("VEYLO_MEMORY", "memory"),
    ]:
        raw = env.get(env_key)
        if raw == "false":
            features[feature_key] = False
        elif raw == "true":
            features[feature_key] = True

    hitl = env.get("VEYLO_HITL")
    if hitl in {"always", "auto", "never"}:
        policy["hitl_mode"] = hitl

    return result


def _deep_merge(target: dict[str, Any], source: dict[str, Any]) -> dict[str, Any]:
    result = deepcopy(target)
    for key, value in source.items():
        if value is None:
            continue
        old = result.get(key)
        if isinstance(old, dict) and isinstance(value, dict):
            result[key] = _deep_merge(old, value)
        else:
            result[key] = deepcopy(value)
    return result


def _config_to_dict(config: VeyloConfig) -> dict[str, Any]:
    return asdict(config)


def _dict_to_config(data: dict[str, Any]) -> VeyloConfig:
    return VeyloConfig(
        llm=LlmConfig(**data.get("llm", {})),
        render_mode=data.get("render_mode", "inline"),
        tools=ToolsConfig(**data.get("tools", {})),
        mcp=McpConfig(**data.get("mcp", {})),
        memory=MemoryConfig(**data.get("memory", {})),
        policy=PolicyConfig(**data.get("policy", {})),
        prompt=PromptConfig(**data.get("prompt", {})),
        features=FeatureConfig(**data.get("features", {})),
        checkpoint=CheckpointConfig(**data.get("checkpoint", {})),
    )


def _expand_home(path: str) -> str:
    return str(Path(path).expanduser())
