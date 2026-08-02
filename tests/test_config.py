import pytest
from pydantic import SecretStr, ValidationError

from furniture_ai.config import AppConfig


def test_production_requires_strong_secret() -> None:
    with pytest.raises(ValidationError):
        AppConfig(environment="production", jwt_secret=SecretStr("short"))


def test_cors_wildcard_rejected_when_credentials_enabled() -> None:
    with pytest.raises(ValidationError):
        AppConfig(allow_origins=["*"], allow_credentials=True)


def test_secure_local_defaults() -> None:
    config = AppConfig()
    assert config.issue_demo_tokens is False
    assert config.allow_credentials is False
    assert "*" not in config.allow_origins
