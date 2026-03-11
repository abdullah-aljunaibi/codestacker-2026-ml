"""Project configuration for data loading, OCR, and training."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class OCRConfig(BaseModel):
    """OCR runtime settings."""

    model_config = ConfigDict(frozen=True)

    tesseract_config: str = "--psm 6"


class PreprocessingConfig(BaseModel):
    """Image preprocessing settings shared by OCR entry points."""

    model_config = ConfigDict(frozen=True)

    grayscale: bool = True
    min_width: int = Field(default=300, ge=1)


class TrainingConfig(BaseModel):
    """Training defaults for the anomaly detector."""

    model_config = ConfigDict(frozen=True)

    anomaly_threshold: float = Field(default=0.45, ge=0.0, le=1.0)
    gradient_boosting_estimators: int = Field(default=150, ge=1)
    gradient_boosting_learning_rate: float = Field(default=0.05, gt=0.0)
    gradient_boosting_max_depth: int = Field(default=3, ge=1)
    gradient_boosting_min_samples_leaf: int = Field(default=2, ge=1)
    random_state: int = 42


class DataConfig(BaseModel):
    """Dataset file naming conventions."""

    model_config = ConfigDict(frozen=True)

    train_file_name: str = "train.jsonl"
    test_file_name: str = "test.jsonl"
    labels_file_name: str = "labels.jsonl"
    stats_file_name: str = "stats.json"
    anomaly_model_file_name: str = "anomaly_model.pkl"


class AppConfig(BaseModel):
    """Top-level immutable application configuration."""

    model_config = ConfigDict(frozen=True)

    data: DataConfig = DataConfig()
    ocr: OCRConfig = OCRConfig()
    preprocessing: PreprocessingConfig = PreprocessingConfig()
    training: TrainingConfig = TrainingConfig()


DEFAULT_CONFIG = AppConfig()
