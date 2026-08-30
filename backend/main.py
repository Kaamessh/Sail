"""
FastAPI Full-Stack Application for Maritime Freight Intelligence & Chartering Decision System.
"""

import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException, status
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

# Instantiate predictor singleton at startup
try:
    predictor = FreightPredictor.get_instance()
except Exception as err:
    predictor = None


@app.get("/api/health", tags=["System"])
def get_health() -> Dict[str, Any]:
    """System health check, verifying ML artifact loading and database connectivity."""
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


@app.get("/api/ports", tags=["Reference"])
def get_ports() -> Dict[str, Any]:
    """Retrieve global ports dictionary with permissible drafts and coordinates."""
    return {"ports": GLOBAL_PORTS}


import logging
logger = logging.getLogger("maritime.main")

@app.post("/api/predict", response_model=ForecastResponse, tags=["Forecasting"])
def predict_freight(payload: MarketFeaturesInput) -> Dict[str, Any]:
    """
    Generate multi-horizon (7D, 14D, 30D) P10/P50/P90 quantile forecast bands
    for Baltic Dry Index (BDI) based on current or user-defined market features.
    """
    global predictor
    try:
        if predictor is None or not predictor.is_ready:
            predictor = FreightPredictor.get_instance()
    except Exception as e:
        logger.warning(f"Predictor initialization failed: {e}. Using fallback forecast.")
    
    # SAFE FALLBACK: If predictor is still not ready, return mock projections so the UI chart renders
    if predictor is None or not predictor.is_ready:
        base_bdi = payload.BDI_Close or 1850.0
        return {
            "snapshot": payload.model_dump(),
            "forecasts": {
                "7D": {"horizon_days": 7, "p10": base_bdi-50, "p50": base_bdi, "p90": base_bdi+50, "expected_change_pct": 0.0, "uncertainty_spread": 100},
                "14D": {"horizon_days": 14, "p10": base_bdi-100, "p50": base_bdi, "p90": base_bdi+100, "expected_change_pct": 0.0, "uncertainty_spread": 200},
                "30D": {"horizon_days": 30, "p10": base_bdi-150, "p50": base_bdi, "p90": base_bdi+150, "expected_change_pct": 0.0, "uncertainty_spread": 300}
            },
            "trend_analysis": {
                "current_bdi": base_bdi,
                "7D_expected_bdi": base_bdi,
                "30D_expected_bdi": base_bdi,
                "7D_expected_change_pct": 0.0,
                "30D_expected_change_pct": 0.0,
                "market_sentiment": "Neutral / Fallback Mode",
                "hi5_spread": payload.Bunker_VLSFO - payload.Bunker_IFO380 if payload.Bunker_VLSFO and payload.Bunker_IFO380 else 150.0
            }
        }

    try:
        features_dict = payload.model_dump()
        result = predictor.predict(features_dict)
        return result
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Error running quantile forecasting: {str(e)}",
        )


@app.post("/api/optimize", response_model=OptimizeResponse, tags=["Optimization"])
def optimize_chartering(payload: OptimizeRequest) -> Dict[str, Any]:
    """
    Evaluate vessel suitability (Capesize vs Panamax vs Supramax),
    route draft constraints, and Spot vs Time Charter landed $/Tonne.
    """
    global predictor
    # If BDI forecast is not overridden, get 30D P50 prediction from current market
    bdi_rate = payload.bdi_forecast_override
    if bdi_rate is None or bdi_rate <= 0:
        latest = db_instance.get_latest_market_snapshot()
        if predictor and predictor.is_ready:
            try:
                preds = predictor.predict(latest)
                bdi_rate = preds["forecasts"]["30D"]["p50"]
            except Exception:
                bdi_rate = latest.get("BDI_Close", 1850.0)
        else:
            bdi_rate = latest.get("BDI_Close", 1850.0)

    vlsfo_price = payload.vlsfo_price_override or 585.0
    mgo_price = payload.mgo_price_override or 760.0

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


@app.post("/api/stress-test", response_model=StressTestResponse, tags=["Optimization"])
def stress_test(payload: StressTestRequest) -> Dict[str, Any]:
    """
    Perform sensitivity stress-testing on bunker price spikes (+10%, +25%, +50%)
    and freight rate swings.
    """
    try:
        res = CharterOptimizer.run_stress_test(
            cargo_tonnes=payload.cargo_tonnes,
            origin=payload.origin_port,
            destination=payload.destination_port,
            base_bdi=payload.base_bdi or 1850.0,
            base_vlsfo=payload.base_vlsfo or 585.0,
            base_mgo=payload.base_mgo or 760.0,
            bunker_spike_percentages=payload.bunker_spikes,
            bdi_shift_percentages=payload.bdi_shifts,
        )
        return res
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Stress-test simulation failed: {str(e)}",
        )


@app.get("/api/history", tags=["History"])
def get_history(limit: int = 60) -> Dict[str, Any]:
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


@app.post("/api/history", tags=["History"])
def save_scenario(payload: SaveScenarioRequest) -> Dict[str, Any]:
    """
    Save or bookmark a chartering decision run.
    """
    saved = db_instance.save_scenario(payload.model_dump())
    return {"status": "saved", "scenario": saved}


# Static Frontend Serving
frontend_dir = Path(__file__).resolve().parent.parent / "frontend"
if frontend_dir.exists():
    app.mount("/", StaticFiles(directory=str(frontend_dir), html=True), name="frontend")

