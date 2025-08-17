
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import BaseModel
from typing import List

class SegmenterConfig(BaseModel):
    num_classes: int = 5  # 0=bg,1=wall,2=door,3=window,4=room
    img_size: int = 512
    batch_size: int = 4
    epochs: int = 30
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    data_root: str = "data/seg"
    models_dir: str = "models/segmenter"

class DetectorConfig(BaseModel):
    data_root: str = "data/detector"
    train_ann: str = "annotations/train.json"
    val_ann: str = "annotations/val.json"
    dataset_slug: str | None = None
    img_size: int = 640
    batch_size: int = 16
    epochs: int = 50
    learning_rate: float = 1e-3
    model: str = "yolov8n.pt"
    models_dir: str = "models/detector"

class AppConfig(BaseModel):
    jwt_secret: str = "please-change-me"
    allow_origins: List[str] = ["*"]

class Settings(BaseSettings):
    segmenter: SegmenterConfig = SegmenterConfig()
    detector: DetectorConfig = DetectorConfig()
    app: AppConfig = AppConfig()
    model_config = SettingsConfigDict(env_file=".env", env_nested_delimiter="__")

settings = Settings()
