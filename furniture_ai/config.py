from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class SegmenterConfig(BaseModel):
    num_classes: int = Field(default=5, ge=2)
    img_size: int = Field(default=512, ge=128, le=2048)
    batch_size: int = Field(default=4, ge=1)
    epochs: int = Field(default=30, ge=1)
    learning_rate: float = Field(default=1e-3, gt=0)
    weight_decay: float = Field(default=1e-4, ge=0)
    data_root: str = "data/seg"
    models_dir: str = "models/segmenter"
    weights_path: str = "models/segmenter/best.pt"
    allow_untrained: bool = False


class DetectorConfig(BaseModel):
    data_root: str = "data/detector"
    train_ann: str = "annotations/train.json"
    val_ann: str = "annotations/val.json"
    dataset_slug: str | None = None
    img_size: int = Field(default=640, ge=128, le=2048)
    batch_size: int = Field(default=16, ge=1)
    epochs: int = Field(default=50, ge=1)
    learning_rate: float = Field(default=1e-3, gt=0)
    model: str = "yolov8n.pt"
    models_dir: str = "models/detector"
    weights_path: str | None = None


class AppConfig(BaseModel):
    environment: Literal["development", "test", "production"] = "development"
    jwt_secret: SecretStr | None = None
    jwt_issuer: str = "floorplan-furnisher-pro"
    jwt_audience: str = "floorplan-furnisher-api"
    token_ttl_minutes: int = Field(default=60, ge=5, le=1440)
    issue_demo_tokens: bool = False
    allow_origins: list[str] = Field(
        default_factory=lambda: ["http://localhost:8501", "http://127.0.0.1:8501"]
    )
    allow_credentials: bool = False
    max_upload_bytes: int = Field(default=10 * 1024 * 1024, ge=1024)
    max_image_pixels: int = Field(default=25_000_000, ge=1_000_000)

    @model_validator(mode="after")
    def validate_security(self) -> AppConfig:
        secret = self.jwt_secret.get_secret_value() if self.jwt_secret else ""
        if self.environment == "production" and len(secret) < 32:
            raise ValueError("APP__JWT_SECRET must contain at least 32 characters in production")
        if self.allow_credentials and "*" in self.allow_origins:
            raise ValueError("Wildcard CORS cannot be used with credentials")
        return self


class Settings(BaseSettings):
    segmenter: SegmenterConfig = Field(default_factory=SegmenterConfig)
    detector: DetectorConfig = Field(default_factory=DetectorConfig)
    app: AppConfig = Field(default_factory=AppConfig)
    model_config = SettingsConfigDict(
        env_file=".env",
        env_nested_delimiter="__",
        extra="ignore",
    )


settings = Settings()
