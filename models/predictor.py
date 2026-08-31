"""
Freight Market Multi-Horizon Quantile Prediction Engine.
Loads pre-trained Gradient Boosting Quantile Regressors for Baltic Dry Index (BDI)
forecasting across 7D, 14D, and 30D horizons at P10, P50, and P90 confidence levels.
Zero heavy top-level dependencies (Lazy Loaded on demand).
"""

import json
import logging
import os
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from dotenv import load_dotenv

load_dotenv()

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
STORAGE_BUCKET = "model-artifacts"


class FreightPredictor:
    """
    Singleton class for loading and executing multi-horizon Baltic Dry Index (BDI)
    quantile regression forecasts. Supports lazy loading and Supabase Storage artifact downloads.
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

    def _sync_artifacts_from_supabase(self, target_dir: Path) -> bool:
        """
        Download the latest trained model artifacts from Supabase Storage bucket 'model-artifacts'.
        """
        supabase_url = os.getenv("SUPABASE_URL", "").strip()
        supabase_key = (
            os.getenv("SUPABASE_SERVICE_ROLE_KEY", "").strip()
            or os.getenv("SUPABASE_KEY", "").strip()
            or os.getenv("SUPABASE_ANON_KEY", "").strip()
        )

        if not supabase_url or not supabase_key:
            return False

        try:
            from supabase import create_client

            client = create_client(supabase_url, supabase_key)
            target_dir.mkdir(parents=True, exist_ok=True)

            required_files = ["freight_quantile_models.joblib", "freight_scalers.joblib", "metadata.json"]
            for filename in required_files:
                file_dest = target_dir / filename
                logger.info(f"Checking Supabase storage for '{filename}'...")
                file_bytes = client.storage.from_(STORAGE_BUCKET).download(filename)
                if file_bytes and len(file_bytes) > 0:
                    with open(file_dest, "wb") as f:
                        f.write(file_bytes)
                    logger.info(f"Downloaded latest '{filename}' from Supabase Storage ({len(file_bytes)} bytes).")
                else:
                    return False

            return True
        except Exception as e:
            logger.info(f"Supabase Storage artifact sync note: {e}")
            return False

    def _locate_artifact_path(self, filename: str, custom_dir: Optional[Union[str, Path]] = None) -> Path:
        """
        Dynamically search for model artifact files using upward directory traversal
        and recursive workspace walkers for local, containerized, and serverless environments.
        """
        if custom_dir:
            custom_target = Path(custom_dir).resolve() / filename
            if custom_target.exists():
                return custom_target

        searched_paths: List[str] = []

        # 1. Upward hierarchy scan from current module directory
        module_dir = Path(__file__).resolve().parent
        current_scan = module_dir
        for _ in range(5):
            candidates = [
                current_scan / "artifacts" / filename,
                current_scan / "models" / "artifacts" / filename,
                current_scan / "backend" / "artifacts" / filename,
                current_scan / "api" / "artifacts" / filename,
                current_scan / filename,
            ]
            for c in candidates:
                searched_paths.append(str(c))
                if c.exists():
                    logger.info(f"Artifact '{filename}' resolved at: {c.resolve()}")
                    return c.resolve()
            if current_scan.parent == current_scan:
                break
            current_scan = current_scan.parent

        # 2. Recursive scan of working directory
        cwd_dir = Path.cwd().resolve()
        for root, _, files in os.walk(cwd_dir):
            if filename in files:
                found_path = Path(root).resolve() / filename
                logger.info(f"Artifact '{filename}' found via cwd walk at: {found_path}")
                return found_path

        # 3. Serverless Lambda/Vercel standard path check
        serverless_roots = [Path("/var/task"), Path("/tmp")]
        for s_root in serverless_roots:
            if s_root.exists():
                for root, _, files in os.walk(s_root):
                    if filename in files:
                        found_path = Path(root).resolve() / filename
                        logger.info(f"Artifact '{filename}' found in serverless root at: {found_path}")
                        return found_path

        error_msg = (
            f"FATAL: Required model artifact '{filename}' could not be located anywhere in the filesystem.\n"
            f"Searched paths include:\n" + "\n".join(f" - {p}" for p in searched_paths[:10])
        )
        logger.error(error_msg)
        raise FileNotFoundError(error_msg)

    def load_artifacts(self, artifacts_dir: Optional[Union[str, Path]] = None) -> None:
        """
        Load joblib models, scalers, and metadata with zero-latency local resolution first,
        falling back to cloud storage sync only if local bundle files are missing.
        Lazy imports joblib inside this method.
        """
        try:
            import joblib

            models_path: Optional[Path] = None
            scalers_path: Optional[Path] = None
            metadata_path: Optional[Path] = None

            # 1. First priority: Aggressive zero-latency local filesystem resolution
            try:
                models_path = self._locate_artifact_path("freight_quantile_models.joblib", artifacts_dir)
                scalers_path = self._locate_artifact_path("freight_scalers.joblib", artifacts_dir)
                metadata_path = self._locate_artifact_path("metadata.json", artifacts_dir)
            except FileNotFoundError:
                logger.info("Local artifact weights missing in deployment bundle. Attempting Supabase Storage sync...")

            # 2. Second priority: Fallback to Supabase Storage if local artifacts not found
            if not models_path or not scalers_path or not metadata_path:
                download_dir = Path(tempfile.gettempdir()) / "maritime_model_artifacts"
                synced = self._sync_artifacts_from_supabase(download_dir)
                if synced:
                    models_path = self._locate_artifact_path("freight_quantile_models.joblib", download_dir)
                    scalers_path = self._locate_artifact_path("freight_scalers.joblib", download_dir)
                    metadata_path = self._locate_artifact_path("metadata.json", download_dir)

            if not models_path or not scalers_path or not metadata_path:
                raise FileNotFoundError("Could not resolve model artifacts locally or via cloud storage sync.")

            logger.info(f"Loading freight models from: {models_path}")
            self._models = joblib.load(models_path)

            logger.info(f"Loading freight scalers from: {scalers_path}")
            self._scalers = joblib.load(scalers_path)

            with open(metadata_path, "r", encoding="utf-8") as f:
                self._metadata = json.load(f)

            self._is_loaded = True
            logger.info("FreightPredictor ML artifacts loaded successfully.")
        except Exception as e:
            logger.error(f"FATAL: Error loading model artifacts: {e}", exc_info=True)
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
        Compute derived technical indicators and spreads from input market features.
        Zero external ML dependencies.
        """
        features = dict(data)

        if "BDI_Close" not in features or features["BDI_Close"] is None:
            raise ValueError("Required feature 'BDI_Close' is missing from input data.")
        bdi_close = float(features["BDI_Close"])

        vlsfo = float(features.get("Bunker_VLSFO", 585.0))
        ifo380 = float(features.get("Bunker_IFO380", 430.0))

        if "Hi5_Spread" not in features or features["Hi5_Spread"] is None:
            features["Hi5_Spread"] = round(vlsfo - ifo380, 2)

        if "BDI_Open" not in features or features["BDI_Open"] is None:
            features["BDI_Open"] = bdi_close
        if "BDI_High" not in features or features["BDI_High"] is None:
            features["BDI_High"] = max(bdi_close, float(features.get("BDI_Open", bdi_close)))
        if "BDI_Low" not in features or features["BDI_Low"] is None:
            features["BDI_Low"] = min(bdi_close, float(features.get("BDI_Open", bdi_close)))
        if "Bunker_MGO" not in features or features["Bunker_MGO"] is None:
            features["Bunker_MGO"] = round(vlsfo * 1.30, 2)

        if "BDI_7D_MA" not in features or features["BDI_7D_MA"] is None:
            features["BDI_7D_MA"] = bdi_close
        if "BDI_14D_MA" not in features or features["BDI_14D_MA"] is None:
            features["BDI_14D_MA"] = bdi_close
        if "BDI_30D_Vol" not in features or features["BDI_30D_Vol"] is None:
            features["BDI_30D_Vol"] = 28.5

        return {
            "BDI_Close": bdi_close,
            "BDI_Open": float(features["BDI_Open"]),
            "BDI_High": float(features["BDI_High"]),
            "BDI_Low": float(features["BDI_Low"]),
            "Bunker_VLSFO": vlsfo,
            "Bunker_MGO": float(features["Bunker_MGO"]),
            "Bunker_IFO380": ifo380,
            "Hi5_Spread": float(features["Hi5_Spread"]),
            "BDI_7D_MA": float(features["BDI_7D_MA"]),
            "BDI_14D_MA": float(features["BDI_14D_MA"]),
            "BDI_30D_Vol": float(features["BDI_30D_Vol"]),
        }

    def predict(self, raw_input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute multi-horizon quantile predictions for Baltic Dry Index.
        Returns P10, P50, and P90 forecast bands.
        Lazy loads pandas and numpy inside this method.
        """
        if not self.is_ready:
            raise RuntimeError("FreightPredictor models are not loaded.")

        import numpy as np
        import pandas as pd

        features_dict = self.compute_derived_features(raw_input_data)
        input_vector = [features_dict[col] for col in FEATURE_COLUMNS]

        X_df = pd.DataFrame([input_vector], columns=FEATURE_COLUMNS)
        scaler = self._scalers.get("features") or self._scalers.get("scaler_7d")
        X_scaled = scaler.transform(X_df)

        forecasts: Dict[str, Dict[str, Any]] = {}
        base_bdi = features_dict["BDI_Close"]

        for h in HORIZONS:
            h_key = f"{h}D"
            models_for_h = self._models[h]

            p10_pred = float(models_for_h["p10"].predict(X_scaled)[0])
            p50_pred = float(models_for_h["p50"].predict(X_scaled)[0])
            p90_pred = float(models_for_h["p90"].predict(X_scaled)[0])

            # Monotonicity adjustment
            p10_val = min(p10_pred, p50_pred, p90_pred)
            p90_val = max(p10_pred, p50_pred, p90_pred)
            p50_val = sorted([p10_pred, p50_pred, p90_pred])[1]

            uncertainty = round(p90_val - p10_val, 2)
            expected_change = round(((p50_val - base_bdi) / base_bdi) * 100.0, 2)

            forecasts[h_key] = {
                "horizon_days": h,
                "p10": round(p10_val, 2),
                "p50": round(p50_val, 2),
                "p90": round(p90_val, 2),
                "uncertainty_spread": uncertainty,
                "expected_change_pct": expected_change,
            }

        # Trend & sentiment summary
        p50_30d = forecasts["30D"]["p50"]
        delta_30d = ((p50_30d - base_bdi) / base_bdi) * 100.0
        sentiment = "Bullish" if delta_30d > 2.0 else ("Bearish" if delta_30d < -2.0 else "Neutral")

        trend_analysis = {
            "current_bdi": base_bdi,
            "expected_bdi_7d": forecasts["7D"]["p50"],
            "expected_bdi_30d": p50_30d,
            "expected_change_7d_pct": forecasts["7D"]["expected_change_pct"],
            "expected_change_30d_pct": delta_30d,
            "market_sentiment": sentiment,
            "hi5_spread": features_dict["Hi5_Spread"],
        }

        return {
            "snapshot": features_dict,
            "forecasts": forecasts,
            "trend_analysis": trend_analysis,
        }
