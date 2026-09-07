from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import yaml

from aico.platform.errors import ConfigurationError

DEFAULT_CONFIG_PATH = Path("config/model-routing.yaml")
DEFAULT_EMBEDDING_DIMENSIONS = 1536


def load_dotenv() -> None:
    """Load KEY=VALUE pairs from .env without overriding variables already in the environment."""
    candidates = [
        Path.cwd() / ".env",
        Path(__file__).resolve().parents[3] / ".env",
    ]
    seen: set[Path] = set()
    for path in candidates:
        path = path.resolve()
        if path in seen or not path.is_file():
            continue
        seen.add(path)
        for raw in path.read_text(encoding="utf-8").splitlines():
            line = raw.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            key = key.strip()
            value = value.strip().strip("'").strip('"')
            if key and key not in os.environ:
                os.environ[key] = value


@dataclass(frozen=True)
class RetrySettings:
    max_attempts: int
    base_delay_ms: int
    max_delay_ms: int
    jitter: bool


@dataclass(frozen=True)
class RouteSettings:
    provider: str
    region: str
    data_boundary: str
    risk_class: str


@dataclass(frozen=True)
class CompatibilityChecks:
    provider: bool
    region: bool
    data_boundary: bool
    risk: bool
    budget: bool


@dataclass(frozen=True)
class GatewayConfig:
    chat_alias: str
    embedding_alias: str
    endpoint_env: str
    endpoint: str
    timeout_seconds: float
    retry: RetrySettings
    max_input_tokens: int
    max_output_tokens: int
    max_items_per_call: int
    primary: RouteSettings
    fallback_enabled: bool
    fallback: RouteSettings
    require_compatibility: CompatibilityChecks
    embedding_dimensions: int = DEFAULT_EMBEDDING_DIMENSIONS
    budget_available: bool = True

    @property
    def api_version(self) -> str:
        return os.environ.get("AICO_FOUNDRY_API_VERSION", "2024-02-01")


def testing_config(
    *,
    embedding_alias: str = "fake-embed-v1",
    chat_alias: str = "fake-chat-v1",
    embedding_dimensions: int = 8,
    timeout_seconds: float = 20.0,
    max_items_per_call: int = 32,
    max_attempts: int = 3,
    base_delay_ms: int = 250,
    max_delay_ms: int = 2000,
    jitter: bool = False,
    fallback_enabled: bool = False,
    primary_provider: str = "microsoft-foundry",
    primary_region: str = "test-region",
    primary_data_boundary: str = "test-boundary",
    primary_risk_class: str = "standard",
    fallback_provider: str = "microsoft-foundry",
    fallback_region: str = "test-region",
    fallback_data_boundary: str = "test-boundary",
    fallback_risk_class: str = "standard",
    budget_available: bool = True,
) -> GatewayConfig:
    """In-memory config for fake-transport tests. Does not read files or the environment."""
    return GatewayConfig(
        chat_alias=chat_alias,
        embedding_alias=embedding_alias,
        endpoint_env="AICO_FOUNDRY_ENDPOINT",
        endpoint="",
        timeout_seconds=timeout_seconds,
        retry=RetrySettings(
            max_attempts=max_attempts,
            base_delay_ms=base_delay_ms,
            max_delay_ms=max_delay_ms,
            jitter=jitter,
        ),
        max_input_tokens=8000,
        max_output_tokens=1000,
        max_items_per_call=max_items_per_call,
        primary=RouteSettings(
            provider=primary_provider,
            region=primary_region,
            data_boundary=primary_data_boundary,
            risk_class=primary_risk_class,
        ),
        fallback_enabled=fallback_enabled,
        fallback=RouteSettings(
            provider=fallback_provider,
            region=fallback_region,
            data_boundary=fallback_data_boundary,
            risk_class=fallback_risk_class,
        ),
        require_compatibility=CompatibilityChecks(
            provider=True,
            region=True,
            data_boundary=True,
            risk=True,
            budget=True,
        ),
        embedding_dimensions=embedding_dimensions,
        budget_available=budget_available,
    )


def config_path() -> Path:
    here = Path(__file__).resolve().parents[3] / DEFAULT_CONFIG_PATH
    cwd = Path.cwd() / DEFAULT_CONFIG_PATH
    if cwd.is_file():
        return cwd
    return here


def load_config(path: Path | None = None) -> GatewayConfig:
    load_dotenv()
    resolved = Path(path) if path is not None else config_path()
    if not resolved.is_file():
        raise ConfigurationError(f"missing routing configuration at {resolved}")
    try:
        payload = yaml.safe_load(resolved.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ConfigurationError(f"invalid routing configuration YAML in {resolved}") from exc
    if not isinstance(payload, dict):
        raise ConfigurationError(f"routing configuration in {resolved} must be a mapping")
    return _parse(payload, source=str(resolved))


def _require(mapping: dict, *keys: str, source: str) -> dict:
    current: object = mapping
    trail = []
    for key in keys:
        trail.append(key)
        if not isinstance(current, dict) or key not in current:
            raise ConfigurationError(
                f"missing required routing field {'.'.join(trail)} in {source}"
            )
        current = current[key]
    return current if isinstance(current, dict) else {"_value": current}


def _value(mapping: dict, *keys: str, source: str) -> object:
    nested = _require(mapping, *keys, source=source)
    if "_value" in nested and len(nested) == 1:
        return nested["_value"]
    return nested


def _parse(payload: dict, *, source: str) -> GatewayConfig:
    endpoint_env = str(_value(payload, "foundry", "endpoint_env", source=source))
    chat_alias = str(_value(payload, "models", "chat", "alias", source=source)).strip()
    embedding_alias = str(_value(payload, "models", "embedding", "alias", source=source)).strip()
    if not chat_alias or not embedding_alias:
        raise ConfigurationError(f"chat and embedding aliases must be non-empty in {source}")

    retry_payload = _require(payload, "resilience", "retry", source=source)
    primary = _require(payload, "routing", "primary", source=source)
    fallback = _require(payload, "routing", "fallback", source=source)
    checks = _require(payload, "routing", "fallback", "require_compatibility", source=source)
    chat_budget = _require(payload, "budgets", "chat", source=source)
    embed_budget = _require(payload, "budgets", "embedding", source=source)

    endpoint = (os.environ.get(endpoint_env) or "").rstrip("/")
    dimensions = int(
        os.environ.get("AZURE_EMBEDDING_DIMENSIONS", str(DEFAULT_EMBEDDING_DIMENSIONS))
    )

    return GatewayConfig(
        chat_alias=chat_alias,
        embedding_alias=embedding_alias,
        endpoint_env=endpoint_env,
        endpoint=endpoint,
        timeout_seconds=float(_value(payload, "resilience", "timeout_seconds", source=source)),
        retry=RetrySettings(
            max_attempts=int(retry_payload["max_attempts"]),
            base_delay_ms=int(retry_payload["base_delay_ms"]),
            max_delay_ms=int(retry_payload["max_delay_ms"]),
            jitter=bool(retry_payload["jitter"]),
        ),
        max_input_tokens=int(chat_budget["max_input_tokens"]),
        max_output_tokens=int(chat_budget["max_output_tokens"]),
        max_items_per_call=int(embed_budget["max_items_per_call"]),
        primary=RouteSettings(
            provider=str(primary["provider"]),
            region=str(primary["region"]),
            data_boundary=str(primary["data_boundary"]),
            risk_class=str(primary["risk_class"]),
        ),
        fallback_enabled=bool(fallback["enabled"]),
        fallback=RouteSettings(
            provider=str(fallback["provider"]),
            region=str(fallback["region"]),
            data_boundary=str(fallback["data_boundary"]),
            risk_class=str(fallback["risk_class"]),
        ),
        require_compatibility=CompatibilityChecks(
            provider=bool(checks["provider"]),
            region=bool(checks["region"]),
            data_boundary=bool(checks["data_boundary"]),
            risk=bool(checks["risk"]),
            budget=bool(checks["budget"]),
        ),
        embedding_dimensions=dimensions,
        budget_available=True,
    )
