"""
Pydantic Request and Response Schemas for Maritime Freight Intelligence API.
"""

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class MarketFeaturesInput(BaseModel):
    BDI_Close: float = Field(..., description="Latest Baltic Dry Index close price", ge=100.0, le=15000.0)
    BDI_Open: Optional[float] = Field(None, description="BDI opening price")
    BDI_High: Optional[float] = Field(None, description="BDI daily high")
    BDI_Low: Optional[float] = Field(None, description="BDI daily low")
    Bunker_VLSFO: float = Field(default=585.0, description="Very Low Sulfur Fuel Oil spot price ($/MT)", ge=100.0, le=2500.0)
    Bunker_MGO: Optional[float] = Field(default=760.0, description="Marine Gas Oil spot price ($/MT)", ge=150.0, le=3500.0)
    Bunker_IFO380: Optional[float] = Field(default=430.0, description="High Sulfur Fuel Oil IFO380 spot price ($/MT)", ge=50.0, le=2000.0)
    Hi5_Spread: Optional[float] = Field(None, description="VLSFO - IFO380 spread ($/MT)")
    BDI_7D_MA: Optional[float] = Field(None, description="7-day moving average of BDI")
    BDI_14D_MA: Optional[float] = Field(None, description="14-day moving average of BDI")
    BDI_30D_Vol: Optional[float] = Field(None, description="30-day historical rolling volatility of BDI")


class HorizonForecastDetail(BaseModel):
    horizon_days: int
    p10: float = Field(..., description="P10 Lower bound forecast (10th percentile)")
    p50: float = Field(..., description="P50 Median expected forecast")
    p90: float = Field(..., description="P90 Upper bound forecast (90th percentile)")
    uncertainty_spread: float
    expected_change_pct: float


class TrendAnalysis(BaseModel):
    current_bdi: float
    expected_bdi_7d: Optional[float] = None
    expected_bdi_30d: Optional[float] = None
    expected_change_7d_pct: Optional[float] = None
    expected_change_30d_pct: Optional[float] = None
    market_sentiment: str
    hi5_spread: float


class ForecastResponse(BaseModel):
    snapshot: Dict[str, float]
    forecasts: Dict[str, HorizonForecastDetail]
    trend_analysis: Dict[str, Any]


class OptimizeRequest(BaseModel):
    cargo_tonnes: float = Field(..., description="Cargo volume to lift in Metric Tonnes", ge=5000.0, le=500000.0)
    origin_port: str = Field(..., description="Origin / Loading Port Name")
    destination_port: str = Field(..., description="Destination / Discharge Port Name")
    laycan_start: Optional[str] = Field(None, description="Laycan window start date (YYYY-MM-DD)")
    laycan_end: Optional[str] = Field(None, description="Laycan window end date (YYYY-MM-DD)")
    custom_max_draft: Optional[float] = Field(None, description="Custom permissible draft restriction (meters)", ge=5.0, le=30.0)
    bdi_forecast_override: Optional[float] = Field(None, description="Optional custom BDI rate override")
    vlsfo_price_override: Optional[float] = Field(None, description="Optional custom VLSFO $/MT price override")
    mgo_price_override: Optional[float] = Field(None, description="Optional custom MGO $/MT price override")
    charter_mode: Optional[str] = Field(default="TimeCharter", description="TimeCharter vs Spot")


class VesselEvaluation(BaseModel):
    vessel_class: str
    dwt: float
    capacity_cargo_mt: float
    laden_draft_m: float
    port_draft_limit_m: float
    draft_clearance_m: float
    is_feasible: bool
    feasibility_status: str
    feasibility_reason: str
    num_voyages: int
    cargo_tonnes: float
    distance_nm: float
    duration_days: float
    sea_days: float
    port_days: float
    daily_hire_rate_usd: float
    charter_hire_cost_usd: float
    vlsfo_consumption_mt: float
    mgo_consumption_mt: float
    fuel_cost_usd: float
    port_dues_usd: float
    canal_costs_usd: float
    total_landed_cost_usd: float
    landed_cost_per_tonne: float
    spot_voyage_rate_pmt: float
    co2_emissions_mt: float


class OptimizeResponse(BaseModel):
    cargo_tonnes: float
    origin_port: str
    destination_port: str
    distance_nm: float
    bdi_rate_used: float
    vlsfo_price_used: float
    mgo_price_used: float
    optimization_status: str
    recommended_vessel: str
    optimal_landed_cost_pmt: float
    total_optimal_cost_usd: float
    recommendation_summary: str
    vessel_options: List[VesselEvaluation]


class StressTestRequest(BaseModel):
    cargo_tonnes: float = Field(default=80000.0, ge=10000.0, le=350000.0)
    origin_port: str = Field(default="Port Hedland")
    destination_port: str = Field(default="Qingdao")
    base_bdi: Optional[float] = 1850.0
    base_vlsfo: Optional[float] = 585.0
    base_mgo: Optional[float] = 760.0
    bunker_spikes: Optional[List[float]] = [0.0, 10.0, 25.0, 50.0]
    bdi_shifts: Optional[List[float]] = [-20.0, 0.0, 20.0]


class StressTestResponse(BaseModel):
    baseline: Dict[str, Any]
    stress_matrix: List[Dict[str, Any]]


class SaveScenarioRequest(BaseModel):
    title: str = Field(..., min_length=3)
    cargo_tonnes: float
    origin: str
    destination: str
    bdi_rate: float
    vlsfo_price: float
    recommended_vessel: str
    optimal_landed_pmt: float
    total_cost_usd: float
    notes: Optional[str] = None
