"""
Freight Market Multi-Horizon Quantile Prediction Engine.
Loads pre-trained Gradient Boosting Quantile Regressors for Baltic Dry Index (BDI)
forecasting across 7D, 14D, and 30D horizons at P10, P50, and P90 confidence levels.
"""

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import joblib
import numpy as np
import pandas as pd

logger = logging.getLogger("maritime.predictor")
logging.basicConfig(level=logging.INFO)

FEATURE_COLUMNS = [
    "BDI_Close",
    "BDI_Open",
    "BDI_High",
    "BDI_Low",
    "Bunker_VLSFO",
    "Bunker_MGO",
    "Bunker_IFO380",
    "Hi5_Spread",
    "BDI_7D_MA",
    "BDI_14D_MA",
    "BDI_30D_Vol",
]

HORIZONS = [7, 14, 30]
QUANTILES = ["p10", "p50", "p90"]


class FreightPredictor:
    """
    Singleton class for loading and executing multi-horizon Baltic Dry Index (BDI)
    quantile regression forecasts.
    """

    _instance: Optional["FreightPredictor"] = None
    _models: Optional[Dict[int, Dict[str, Any]]] = None
    _scalers: Optional[Dict[str, Any]] = None
    _metadata: Optional[Dict[str, Any]] = None
    _is_loaded: bool = False

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(FreightPredictor, cls).__new__(cls)
        return cls._instance

    def __init__(self, artifacts_dir: Optional[Union[str, Path]] = None):
        if not self._is_loaded:
            self.load_artifacts(artifacts_dir)

    @classmethod
    def get_instance(cls, artifacts_dir: Optional[Union[str, Path]] = None) -> "FreightPredictor":
        if cls._instance is None or not cls._instance._is_loaded:
            cls._instance = cls(artifacts_dir)
        return cls._instance

    def _locate_artifact_path(self, filename: str, custom_dir: Optional[Union[str, Path]] = None) -> Path:
        """
        Aggressively resolve the absolute path to model artifacts across local,
        packaged, and serverless environments.
        """
        candidate_paths: List[Path] = []
        if custom_dir:
            candidate_paths.append(Path(custom_dir).resolve() / filename)

        module_dir = Path(__file__).resolve().parent
        root_dir = module_dir.parent
        cwd_dir = Path.cwd().resolve()

        candidate_paths.extend([
            module_dir / "artifacts" / filename,
            root_dir / "models" / "artifacts" / filename,
            cwd_dir / "models" / "artifacts" / filename,
            cwd_dir / "artifacts" / filename,
            Path("/var/task/models/artifacts") / filename,
            Path("/var/task/artifacts") / filename,
        ])

        for target in candidate_paths:
            if target.exists():
                return target

        checked_str = "\n - ".join([str(p) for p in candidate_paths])
        raise FileNotFoundError(
            f"CRITICAL: Model artifact '{filename}' could not be found. Checked paths:\n - {checked_str}"
        )

    def load_artifacts(self, artifacts_dir: Optional[Union[str, Path]] = None) -> None:
        """Load joblib models, scalers, and metadata."""
        try:
            models_path = self._locate_artifact_path("freight_quantile_models.joblib", artifacts_dir)
            scalers_path = self._locate_artifact_path("freight_scalers.joblib", artifacts_dir)
            metadata_path = self._locate_artifact_path("metadata.json", artifacts_dir)

            logger.info(f"Loading freight models from: {models_path}")
            self._models = joblib.load(models_path)

            logger.info(f"Loading freight scalers from: {scalers_path}")
            self._scalers = joblib.load(scalers_path)

            with open(metadata_path, "r", encoding="utf-8") as f:
                self._metadata = json.load(f)

            self._is_loaded = True
            logger.info("FreightPredictor ML artifacts loaded successfully.")
        except Exception as e:
            logger.error(f"Error loading model artifacts: {e}", exc_info=True)
            self._is_loaded = False
            raise

    @property
    def is_ready(self) -> bool:
        return self._is_loaded and self._models is not None and self._scalers is not None

    @property
    def metadata(self) -> Dict[str, Any]:
        return self._metadata or {}

    @staticmethod
    def compute_derived_features(data: Dict[str, Any]) -> Dict[str, float]:
        """
        Compute derived technical indicators and spreads if not provided directly.
        Derived features:
        - Hi5_Spread = Bunker_VLSFO - Bunker_IFO380
        - BDI_7D_MA (defaults to BDI_Close if no series provided)
        - BDI_14D_MA (defaults to BDI_Close * 0.99 if no series provided)
        - BDI_30D_Vol (defaults to historical rolling baseline if missing)
        """
        features = dict(data)

        # 1. Hi5 Spread
        vlsfo = float(features.get("Bunker_VLSFO", 585.0))
        ifo380 = float(features.get("Bunker_IFO380", 430.0))
        if "Hi5_Spread" not in features or features["Hi5_Spread"] is None:
            features["Hi5_Spread"] = round(vlsfo - ifo380, 2)

        # 2. BDI Open/High/Low defaults relative to Close if missing
        bdi_close = float(features.get("BDI_Close", 1850.0))
        if "BDI_Open" not in features or features["BDI_Open"] is None:
            features["BDI_Open"] = bdi_close
        if "BDI_High" not in features or features["BDI_High"] is None:
            features["BDI_High"] = max(bdi_close, float(features.get("BDI_Open", bdi_close))) * 1.01
        if "BDI_Low" not in features or features["BDI_Low"] is None:
            features["BDI_Low"] = min(bdi_close, float(features.get("BDI_Open", bdi_close))) * 0.99
        if "Bunker_MGO" not in features or features["Bunker_MGO"] is None:
            features["Bunker_MGO"] = round(vlsfo * 1.30, 2)

        # 3. Moving Averages & Volatility
        if "BDI_7D_MA" not in features or features["BDI_7D_MA"] is None:
            features["BDI_7D_MA"] = round(bdi_close * 0.995, 2)
        if "BDI_14D_MA" not in features or features["BDI_14D_MA"] is None:
            features["BDI_14D_MA"] = round(bdi_close * 0.985, 2)
        if "BDI_30D_Vol" not in features or features["BDI_30D_Vol"] is None:
            features["BDI_30D_Vol"] = round(bdi_close * 0.025, 2)

        return {k: float(features[k]) for k in FEATURE_COLUMNS if k in features}

    def prepare_feature_array(self, features_dict: Dict[str, Any]) -> np.ndarray:
        """
        Validate and format feature dictionary into a 2D numpy array [1, num_features]
        in the strict order expected by the trained scalers.
        """
        complete_features = self.compute_derived_features(features_dict)
        missing_keys = [col for col in FEATURE_COLUMNS if col not in complete_features]
        if missing_keys:
            raise ValueError(f"Missing required feature columns: {missing_keys}")

        feature_values = [complete_features[col] for col in FEATURE_COLUMNS]
        return np.array([feature_values], dtype=np.float64)

    def predict(self, features_dict: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate multi-horizon quantile forecast bands from real ML models.
        """
        if not self.is_ready:
            raise RuntimeError("FreightPredictor models are not loaded.")

        prepared_dict = self.compute_derived_features(features_dict)
        feature_matrix = self.prepare_feature_array(prepared_dict)
        base_bdi = prepared_dict["BDI_Close"]

        results: Dict[str, Any] = {
            "snapshot": prepared_dict,
            "forecasts": {},
        }

        for horizon in HORIZONS:
            scaler_key = f"scaler_{horizon}d"
            if scaler_key not in self._scalers:
                raise KeyError(f"Scaler key '{scaler_key}' not found in loaded scalers.")

            scaler = self._scalers[scaler_key]
            scaled_features = scaler.transform(feature_matrix)

            horizon_models = self._models.get(horizon, {})
            p10 = float(horizon_models["p10"].predict(scaled_features)[0])
            p50 = float(horizon_models["p50"].predict(scaled_features)[0])
            p90 = float(horizon_models["p90"].predict(scaled_features)[0])

            # Monotonic quantile sorting guarantee
            sorted_quantiles = sorted([p10, p50, p90])
            p10_clean = round(sorted_quantiles[0], 2)
            p50_clean = round(sorted_quantiles[1], 2)
            p90_clean = round(sorted_quantiles[2], 2)

            results["forecasts"][f"{horizon}D"] = {
                "horizon_days": horizon,
                "p10": p10_clean,
                "p50": p50_clean,
                "p90": p90_clean,
                "uncertainty_spread": round(p90_clean - p10_clean, 2),
                "expected_change_pct": round(((p50_clean - base_bdi) / max(base_bdi, 1)) * 100, 2),
            }

        # Trend & sentiment metrics
        p50_7d = results["forecasts"]["7D"]["p50"]
        p50_30d = results["forecasts"]["30D"]["p50"]
        chg_7d = results["forecasts"]["7D"]["expected_change_pct"]
        chg_30d = results["forecasts"]["30D"]["expected_change_pct"]

        if chg_30d > 5.0:
            sentiment = "Bullish"
        elif chg_30d < -5.0:
            sentiment = "Bearish"
        else:
            sentiment = "Neutral / Rangebound"

        results["trend_analysis"] = {
            "current_bdi": base_bdi,
            "7D_expected_bdi": p50_7d,
            "30D_expected_bdi": p50_30d,
            "7D_expected_change_pct": chg_7d,
            "30D_expected_change_pct": chg_30d,
            "market_sentiment": sentiment,
            "hi5_spread": prepared_dict["Hi5_Spread"],
        }

        return results
