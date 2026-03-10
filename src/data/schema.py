"""Pydantic schemas for dataset IO."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


FraudType = Literal["none", "price_change", "text_edit", "layout_edit"]


class ExtractedFields(BaseModel):
    """Structured receipt fields."""

    model_config = ConfigDict(extra="ignore")

    vendor: str | None = None
    date: str | None = None
    total: str | None = None

    @field_validator("vendor", "date", "total", mode="before")
    @classmethod
    def _normalize_optional_text(cls, value: object) -> object:
        if value is None:
            return None
        if isinstance(value, str):
            stripped = value.strip()
            return stripped or None
        return value


class DatasetLabel(BaseModel):
    """Forgery label for a dataset record."""

    model_config = ConfigDict(extra="ignore")

    is_forged: int = Field(ge=0, le=1)
    fraud_type: FraudType | str = "none"


class DatasetRecord(BaseModel):
    """A train or test record from the challenge JSONL files."""

    model_config = ConfigDict(extra="ignore")

    id: str = Field(min_length=1)
    image_path: str = Field(min_length=1)
    fields: ExtractedFields = Field(default_factory=ExtractedFields)
    label: DatasetLabel | None = None


class LabelOnlyRecord(BaseModel):
    """A labels.jsonl row used for evaluation datasets."""

    model_config = ConfigDict(extra="ignore")

    id: str = Field(min_length=1)
    label: DatasetLabel


class PredictionRecord(BaseModel):
    """Prediction row written by the submission."""

    model_config = ConfigDict(extra="ignore")

    id: str = Field(min_length=1)
    vendor: str | None = None
    date: str | None = None
    total: str | None = None
    is_forged: int = Field(ge=0, le=1)

    @field_validator("vendor", "date", "total", mode="before")
    @classmethod
    def _normalize_prediction_text(cls, value: object) -> object:
        if value is None:
            return None
        if isinstance(value, str):
            stripped = value.strip()
            return stripped or None
        return value
