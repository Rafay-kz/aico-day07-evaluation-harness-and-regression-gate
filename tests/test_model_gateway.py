"""Task 1 + 2 + 5: gateway boundary, config, identity, metadata, log sanitization."""

from __future__ import annotations

import ast
import logging
from pathlib import Path

import pytest

from aico.platform.config import load_config
from aico.platform.config import testing_config as make_testing_config
from aico.platform.errors import ConfigurationError, OperationCancelled
from aico.platform.fake_transport import FAKE_CHAT_TEXT, FakeTransport
from aico.platform.model_gateway import (
    LOGGER,
    CancellationToken,
    ChatMessage,
    ChatRequest,
    EmbedRequest,
    GatewayMetadata,
    ModelGateway,
    TokenUsage,
    build_fake_gateway,
)
from aico.retrieval.embedding_provider import FakeEmbeddingProvider

SRC_ROOT = Path(__file__).resolve().parents[1] / "src" / "aico"
PLATFORM_ROOT = SRC_ROOT / "platform"
METADATA_FIELDS = {"model_alias", "token_usage", "latency_ms", "retry_count", "budget_status"}
CONTENT_FIELDS = {"prompt", "prompts", "completion", "completions", "messages", "text", "content"}
UNIQUE_PROMPT = "UNIQUE_PROMPT_TOKEN_DO_NOT_LOG"
UNIQUE_HEADER = "Bearer UNIQUE_AUTH_HEADER_DO_NOT_LOG"


def _iter_python_files(root: Path):
    yield from sorted(root.rglob("*.py"))


def _imported_names(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                names.add(alias.name)
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.add(node.module)
    return names


def _is_model_sdk(name: str) -> bool:
    return name == "openai" or name.startswith("openai.") or name.startswith("azure.ai")


def _is_identity_sdk(name: str) -> bool:
    return name == "azure.identity" or name.startswith("azure.identity")


def test_sdk_not_imported_outside_platform() -> None:
    offenders: list[str] = []
    for path in _iter_python_files(SRC_ROOT):
        if PLATFORM_ROOT in path.parents or path.parent == PLATFORM_ROOT:
            continue
        for name in _imported_names(path):
            if _is_model_sdk(name) or _is_identity_sdk(name):
                offenders.append(f"{path.relative_to(SRC_ROOT.parent)} imports {name}")
    assert offenders == []


def test_foundry_adapter_is_the_only_openai_import() -> None:
    openai_files = []
    for path in _iter_python_files(SRC_ROOT):
        if any(_is_model_sdk(name) for name in _imported_names(path)):
            openai_files.append(str(path.relative_to(SRC_ROOT)))
    assert openai_files == ["platform/foundry_adapter.py"]


def test_identity_sdk_only_in_foundry_adapter() -> None:
    files = []
    for path in _iter_python_files(SRC_ROOT):
        if any(_is_identity_sdk(name) for name in _imported_names(path)):
            files.append(str(path.relative_to(SRC_ROOT)))
    assert files == ["platform/foundry_adapter.py"]


def test_foundry_adapter_uses_identity_not_api_keys() -> None:
    source = Path("src/aico/platform/foundry_adapter.py").read_text(encoding="utf-8")
    assert "DefaultAzureCredential" in source
    assert "get_bearer_token_provider" in source
    assert "AZURE_EMBEDDING_API_KEY" not in source
    assert "api-key" not in source.lower()


def test_no_embedded_credentials_in_committed_files() -> None:
    roots = [Path("src"), Path("config")]
    banned = ("BEGIN PRIVATE KEY", "sk-proj-", "client_secret:")
    for root in roots:
        if not root.exists():
            continue
        for path in root.rglob("*"):
            if not path.is_file() or path.suffix in {".pyc", ".png"}:
                continue
            text = path.read_text(encoding="utf-8", errors="ignore")
            for token in banned:
                assert token not in text, f"{path} contains {token}"


def test_embed_through_gateway_returns_vectors_and_metadata() -> None:
    transport = FakeTransport(embed_dimensions=8)
    gateway = ModelGateway(transport, make_testing_config(), sleep=lambda _s: None)
    result = gateway.embed(EmbedRequest(texts=["termination notice period"]))
    assert len(result.vectors) == 1
    assert len(result.vectors[0]) == 8
    _assert_sanitized_metadata(result.metadata, expected_alias="fake-embed-v1")
    assert transport.embed_calls == 1


def test_chat_through_gateway_returns_text_and_metadata() -> None:
    transport = FakeTransport()
    gateway = ModelGateway(transport, make_testing_config(), sleep=lambda _s: None)
    result = gateway.chat(
        ChatRequest(messages=[ChatMessage(role="user", content="ping")])
    )
    assert result.text == FAKE_CHAT_TEXT
    _assert_sanitized_metadata(result.metadata, expected_alias="fake-chat-v1")
    assert transport.chat_calls == 1


def test_metadata_does_not_include_prompt_or_completion() -> None:
    result = build_fake_gateway().chat(
        ChatRequest(messages=[ChatMessage(role="user", content="secret prompt")])
    )
    payload = result.metadata.__dict__
    assert CONTENT_FIELDS.isdisjoint(payload)
    assert "secret prompt" not in str(payload)
    assert FAKE_CHAT_TEXT not in str(payload)
    assert isinstance(result.metadata.token_usage, TokenUsage)
    assert result.metadata.token_usage.input_tokens is None


def test_day2_fake_provider_embeds_through_the_gateway() -> None:
    provider = FakeEmbeddingProvider()
    assert isinstance(provider._gateway, ModelGateway)
    vectors = provider.embed(["insolvent supplier"])
    assert len(vectors) == 1
    assert len(vectors[0]) == provider.dimensions
    assert provider.calls == 1
    again = provider.embed(["insolvent supplier"])
    assert vectors == again


def test_request_timeout_is_passed_to_the_transport() -> None:
    transport = FakeTransport()
    gateway = ModelGateway(transport, make_testing_config(), sleep=lambda _s: None)
    gateway.embed(EmbedRequest(texts=["hello"], timeout_seconds=7.5))
    assert transport.last_timeout_seconds == 7.5


def test_pre_cancelled_token_fails_without_calling_transport() -> None:
    transport = FakeTransport()
    gateway = ModelGateway(transport, make_testing_config(), sleep=lambda _s: None)
    token = CancellationToken()
    token.cancel()
    with pytest.raises(OperationCancelled):
        gateway.embed(EmbedRequest(texts=["hello"], cancellation=token))
    assert transport.embed_calls == 0


def test_routing_yaml_loads_without_secrets() -> None:
    config = load_config()
    assert config.embedding_alias
    assert config.chat_alias
    assert config.endpoint_env == "AICO_FOUNDRY_ENDPOINT"
    assert config.retry.max_attempts >= 1
    raw = Path("config/model-routing.yaml").read_text(encoding="utf-8").lower()
    for banned in ("api_key:", "api-key:", "bearer ", "password:"):
        assert banned not in raw


def test_missing_routing_file_is_configuration_error(tmp_path: Path) -> None:
    with pytest.raises(ConfigurationError, match="missing routing configuration"):
        load_config(tmp_path / "missing.yaml")


def test_invalid_yaml_is_configuration_error(tmp_path: Path) -> None:
    path = tmp_path / "broken.yaml"
    path.write_text("models: [unterminated\n", encoding="utf-8")
    with pytest.raises(ConfigurationError, match="invalid routing configuration YAML"):
        load_config(path)


def test_missing_required_alias_is_configuration_error(tmp_path: Path) -> None:
    path = tmp_path / "partial.yaml"
    path.write_text(
        "\n".join(
            [
                "version: '1.0'",
                "foundry:",
                "  endpoint_env: AICO_FOUNDRY_ENDPOINT",
                "models:",
                "  chat:",
                "    alias: demo-chat",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ConfigurationError, match="missing required routing field"):
        load_config(path)


def test_logs_exclude_prompt_completion_and_secrets(caplog: pytest.LogCaptureFixture) -> None:
    transport = FakeTransport()
    gateway = ModelGateway(transport, make_testing_config(), sleep=lambda _s: None)
    with caplog.at_level(logging.INFO, logger=LOGGER.name):
        gateway.chat(
            ChatRequest(
                messages=[ChatMessage(role="user", content=UNIQUE_PROMPT + " " + UNIQUE_HEADER)]
            )
        )
    text = caplog.text
    assert UNIQUE_PROMPT not in text
    assert UNIQUE_HEADER not in text
    assert FAKE_CHAT_TEXT not in text
    assert "operation=chat" in text
    assert "outcome=success" in text
    assert "model_alias=fake-chat-v1" in text


def _assert_sanitized_metadata(
    metadata: GatewayMetadata,
    *,
    expected_alias: str,
    retry_count: int = 0,
) -> None:
    assert set(metadata.__dict__) == METADATA_FIELDS
    assert metadata.model_alias == expected_alias
    assert metadata.latency_ms >= 0
    assert metadata.retry_count == retry_count
    assert metadata.budget_status == "within_budget"
    assert CONTENT_FIELDS.isdisjoint(metadata.__dict__)
