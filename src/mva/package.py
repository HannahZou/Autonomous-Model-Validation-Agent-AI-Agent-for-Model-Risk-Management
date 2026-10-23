"""Model package: the unit a validator receives.

A model package is a folder containing ``model_package.yaml`` (the manifest)
plus the data files it references. Two model types are supported:

* ``credit_pd``  — a probability-of-default model (binary target)
* ``market_var`` — a Value-at-Risk model on a portfolio of price series
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Literal, Union

import pandas as pd
import yaml
from pydantic import BaseModel, Field, ValidationError, model_validator

MANIFEST_NAME = "model_package.yaml"


class PackageError(Exception):
    """Raised when a model package is missing, malformed or inconsistent."""


class ModelDocumentation(BaseModel):
    purpose: str
    intended_use: str = ""
    owner: str = ""
    assumptions: list[str] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)


class CreditData(BaseModel):
    train: str
    test: str


class CreditPDSpec(BaseModel):
    model_type: Literal["credit_pd"]
    data: CreditData
    target: str
    features: list[str] = Field(min_length=1)
    id_column: str | None = None
    # Declared valid ranges, e.g. {"age": [18, 100]}
    feature_ranges: dict[str, tuple[float, float]] = Field(default_factory=dict)


class MarketData(BaseModel):
    prices: str
    date_column: str = "date"


class VaRConfig(BaseModel):
    method: Literal["historical", "parametric"] = "historical"
    confidence: float = Field(0.99, gt=0.5, lt=1.0)
    window: int = Field(250, ge=20)
    horizon_days: int = Field(1, ge=1)


class MarketVaRSpec(BaseModel):
    model_type: Literal["market_var"]
    data: MarketData
    assets: list[str] = Field(min_length=1)
    weights: dict[str, float]
    var: VaRConfig = Field(default_factory=VaRConfig)

    @model_validator(mode="after")
    def _weights_match_assets(self) -> "MarketVaRSpec":
        missing = set(self.assets) - set(self.weights)
        extra = set(self.weights) - set(self.assets)
        if missing or extra:
            raise ValueError(
                f"weights must cover exactly the listed assets (missing={sorted(missing)}, extra={sorted(extra)})"
            )
        return self


Spec = Annotated[Union[CreditPDSpec, MarketVaRSpec], Field(discriminator="model_type")]


class Manifest(BaseModel):
    model_id: str
    name: str
    version: str = "1.0"
    documentation: ModelDocumentation
    spec: Spec


class ModelPackage:
    def __init__(self, root: Path, manifest: Manifest):
        self.root = root
        self.manifest = manifest
        self._data: dict[str, pd.DataFrame] | None = None

    # -- construction -------------------------------------------------------
    @classmethod
    def load(cls, root: str | Path) -> "ModelPackage":
        root = Path(root)
        path = root / MANIFEST_NAME
        if not path.exists():
            raise PackageError(f"No {MANIFEST_NAME} found in {root}")
        try:
            raw = yaml.safe_load(path.read_text()) or {}
        except yaml.YAMLError as exc:
            raise PackageError(f"Manifest is not valid YAML: {exc}") from exc
        # Allow the spec fields to sit at top level of the YAML for readability.
        spec_keys = {"model_type", "data", "target", "features", "id_column",
                     "feature_ranges", "assets", "weights", "var"}
        raw = dict(raw)
        raw["spec"] = {k: raw.pop(k) for k in list(raw) if k in spec_keys}
        try:
            manifest = Manifest.model_validate(raw)
        except ValidationError as exc:
            raise PackageError(f"Manifest failed validation:\n{exc}") from exc
        return cls(root, manifest)

    # -- convenience --------------------------------------------------------
    @property
    def model_type(self) -> str:
        return self.manifest.spec.model_type

    @property
    def spec(self) -> CreditPDSpec | MarketVaRSpec:
        return self.manifest.spec

    def _read_csv(self, rel: str, **kwargs) -> pd.DataFrame:
        path = self.root / rel
        if not path.exists():
            raise PackageError(f"Data file referenced by manifest not found: {rel}")
        return pd.read_csv(path, **kwargs)

    def load_data(self) -> dict[str, pd.DataFrame]:
        """Return the package's datasets, keyed by role ("train", "test" or "prices")."""
        if self._data is not None:
            return self._data
        spec = self.spec
        if isinstance(spec, CreditPDSpec):
            data = {"train": self._read_csv(spec.data.train), "test": self._read_csv(spec.data.test)}
            needed = set(spec.features) | {spec.target}
            for role, df in data.items():
                missing = sorted(needed - set(df.columns))
                if missing:
                    raise PackageError(f"{role} data is missing declared columns: {missing}")
        else:
            prices = self._read_csv(spec.data.prices)
            needed = set(spec.assets) | {spec.data.date_column}
            missing = sorted(needed - set(prices.columns))
            if missing:
                raise PackageError(f"price data is missing declared columns: {missing}")
            prices[spec.data.date_column] = pd.to_datetime(prices[spec.data.date_column], errors="coerce")
            data = {"prices": prices}
        self._data = data
        return data
