"""
FastAPI Full-Stack Application for Maritime Freight Intelligence & Chartering Decision System.
Strict 100% Genuine ML Pipeline with Autonomous Startup Verification.
"""

import logging
import os
import sys
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from backend.database import db_instance
from backend.optimizer import GLOBAL_PORTS, CharterOptimizer
from backend.schemas import (
    ForecastResponse,
    MarketFeaturesInput,
    OptimizeRequest,
    OptimizeResponse,
    SaveScenarioRequest,
    StressTestRequest,
    StressTestResponse,
)
from models.predictor import FreightPredictor

logger = logging.getLogger("maritime.main")
logging.basicConfig(level=logging.INFO)

app = FastAPI(
    title="Maritime Freight Intelligence & Chartering Decision System",
    description="Multi-horizon Quantile Regression Freight Forecasting, Prescriptive Chartering Optimizer, and Dark Maritime Dashboard",
    version="1.0.0",
)

# Enable CORS for wide compatibility
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global Predictor Reference
try:
    predictor = FreightPredictor.get_instance()
except Exception as err:
    logger.warning(f"Predictor eager init deferral: {err}")
    predictor = None


@app.on_event("startup")
def startup_and_verify_ml_pipeline():
    """
    Autonomous Startup Test:
    Ensures model weights and scalers load properly and executes an actual test prediction.
    If the test fails, immediately terminates execution with an explicit traceback.
    """
    global predictor
    logger.info("Starting Autonomous ML Pipeline Verification...")
    try:
        predictor = FreightPredictor.get_instance()
        if not predictor or not predictor.is_ready:
            raise RuntimeError("FreightPredictor instance failed initialization.")

        # Test inference with latest market record
        test_snapshot = db_instance.get_latest_market_snapshot()
        logger.info(f"Running startup test inference with snapshot: {test_snapshot}")
        test_result = predictor.predict(test_snapshot)

        if "forecasts" not in test_result or "30D" not in test_result["forecasts"]:
            raise ValueError(f"Startup test inference returned unexpected payload: {test_result}")

        logger.info(
            f"STARTUP ML VERIFICATION SUCCESSFUL: 30D P50 Forecast = {test_result['forecasts']['30D']['p50']} pts."
        )
    except Exception as exc:
        err_trace = traceback.format_exc()
        logger.critical(f"FATAL: Autonomous ML Pipeline Verification FAILED on startup!\n{err_trace}")
        print(f"\n==================== FATAL ML STARTUP ERROR ====================\n{err_trace}\n", file=sys.stderr)
        sys.exit(1)


# Router for API Endpoints (mounted both with and without /api prefix for robust Vercel compatibility)
api_router = APIRouter()


@api_router.get("/health", tags=["System"])
def get_health() -> Dict[str, Any]:
    """System health check, verifying ML artifact loading and database connectivity."""
    global predictor
    if predictor is None or not predictor.is_ready:
        try:
            predictor = FreightPredictor.get_instance()
        except Exception as e:
            logger.error(f"Health check predictor reload error: {e}")

    is_ready = predictor is not None and predictor.is_ready
    db_status = db_instance.get_status()

    return {
        "status": "healthy" if is_ready else "degraded",
        "service": "Maritime Freight Intelligence System",
        "model_artifacts_loaded": is_ready,
        "horizons_supported": [7, 14, 30],
        "quantiles_supported": ["p10", "p50", "p90"],
        "database": db_status,
    }


@api_router.get("/ports", tags=["Reference"])
def get_ports() -> Dict[str, Any]:
    """Retrieve global ports dictionary with permissible drafts and coordinates."""
    return {"ports": GLOBAL_PORTS}


@api_router.post("/predict", response_model=ForecastResponse, tags=["Forecasting"])
def predict_freight(payload: MarketFeaturesInput) -> Dict[str, Any]:
    """
    Generate multi-horizon (7D, 14D, 30D) P10/P50/P90 quantile forecast bands
    for Baltic Dry Index (BDI) directly from the trained Gradient Boosting models.
    """
    global predictor
    if predictor is None or not predictor.is_ready:
        try:
            predictor = FreightPredictor.get_instance()
        except Exception as e:
            logger.error(f"Predictor loading error: {traceback.format_exc()}")
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=f"Model predictor failed to initialize: {e}",
            )

    try:
        features_dict = payload.model_dump()
        result = predictor.predict(features_dict)
        return result
    except Exception as e:
        logger.error(f"Predictor inference error: {traceback.format_exc()}")
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Prediction error: {str(e)} | Trace: {traceback.format_exc()}",
        )


@api_router.post("/optimize", response_model=OptimizeResponse, tags=["Optimization"])
def optimize_chartering(payload: OptimizeRequest) -> Dict[str, Any]:
    """
    Evaluate vessel suitability (Capesize vs Panamax vs Supramax),
    route draft constraints, and Spot vs Time Charter landed $/Tonne.
    """
    global predictor
    bdi_rate = payload.bdi_forecast_override
    if bdi_rate is None or bdi_rate <= 0:
        latest = db_instance.get_latest_market_snapshot()
        if predictor and predictor.is_ready:
            preds = predictor.predict(latest)
            bdi_rate = preds["forecasts"]["30D"]["p50"]
        else:
            bdi_rate = float(latest["BDI_Close"])

    vlsfo_price = payload.vlsfo_price_override or float(db_instance.get_latest_market_snapshot().get("Bunker_VLSFO", 585.0))
    mgo_price = payload.mgo_price_override or float(db_instance.get_latest_market_snapshot().get("Bunker_MGO", 760.0))

    try:
        result = CharterOptimizer.optimize_charter(
            cargo_tonnes=payload.cargo_tonnes,
            origin=payload.origin_port,
            destination=payload.destination_port,
            bdi_forecast=bdi_rate,
            vlsfo_price=vlsfo_price,
            mgo_price=mgo_price,
            custom_max_draft=payload.custom_max_draft,
            charter_mode=payload.charter_mode or "TimeCharter",
        )
        return result
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Optimization failed: {str(e)}",
        )


@api_router.post("/stress-test", response_model=StressTestResponse, tags=["Optimization"])
def stress_test(payload: StressTestRequest) -> Dict[str, Any]:
    """
    Perform sensitivity stress-testing on bunker price spikes and freight rate swings.
    """
    latest = db_instance.get_latest_market_snapshot()
    base_bdi = payload.base_bdi or float(latest.get("BDI_Close", 1850.0))
    base_vlsfo = payload.base_vlsfo or float(latest.get("Bunker_VLSFO", 585.0))
    base_mgo = payload.base_mgo or float(latest.get("Bunker_MGO", 760.0))

    try:
        res = CharterOptimizer.run_stress_test(
            cargo_tonnes=payload.cargo_tonnes,
            origin=payload.origin_port,
            destination=payload.destination_port,
            base_bdi=base_bdi,
            base_vlsfo=base_vlsfo,
            base_mgo=base_mgo,
            bunker_spike_percentages=payload.bunker_spikes,
            bdi_shift_percentages=payload.bdi_shifts,
        )
        return res
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Stress-test simulation failed: {str(e)}",
        )


@api_router.get("/history", tags=["History"])
def get_history(limit: int = 45) -> Dict[str, Any]:
    """
    Retrieve historical market snapshots and saved scenarios.
    """
    history_records = db_instance.get_market_history(limit=limit)
    scenarios = db_instance.get_saved_scenarios(limit=20)
    latest_snapshot = db_instance.get_latest_market_snapshot()

    return {
        "count": len(history_records),
        "history": history_records,
        "latest_snapshot": latest_snapshot,
        "saved_scenarios": scenarios,
    }


@api_router.post("/history", tags=["History"])
def save_scenario(payload: SaveScenarioRequest) -> Dict[str, Any]:
    """
    Save or bookmark a chartering decision run.
    """
    saved = db_instance.save_scenario(payload.model_dump())
    return {"status": "saved", "scenario": saved}


# Mount API routes at both /api and root
app.include_router(api_router, prefix="/api")
app.include_router(api_router, prefix="")

# Static Frontend Serving (only for local standalone server)
if not os.getenv("VERCEL") and not os.getenv("AWS_LAMBDA_FUNCTION_NAME"):
    frontend_dir = Path(__file__).resolve().parent.parent / "frontend"
    if frontend_dir.exists():
        app.mount("/", StaticFiles(directory=str(frontend_dir), html=True), name="frontend")
