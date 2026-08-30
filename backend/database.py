"""
Database & Persistence layer supporting Supabase (PostgreSQL)
with fallback in-memory / JSON storage for resilient offline & serverless operation.
"""

import json
import logging
import os
import uuid
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional
from dotenv import load_dotenv

# Load local .env file
load_dotenv()

logger = logging.getLogger("maritime.database")

def get_clean_supabase_url() -> str:
    url = os.getenv("SUPABASE_URL", "").strip()
    if url.endswith("/rest/v1/"):
        url = url[:-9]
    elif url.endswith("/rest/v1"):
        url = url[:-8]
    if url.endswith("/"):
        url = url[:-1]
    return url

SUPABASE_URL = get_clean_supabase_url()
SUPABASE_KEY = (
    os.getenv("SUPABASE_SERVICE_ROLE_KEY", "").strip()
    or os.getenv("SUPABASE_ANON_KEY", "").strip()
    or os.getenv("SUPABASE_KEY", "").strip()
)


def generate_seed_market_history() -> List[Dict[str, Any]]:
    """
    Generate realistic 60-day historical time-series of Baltic Dry Index (BDI),
    Bunker Fuel prices (VLSFO, MGO, IFO380), and Hi5 Spread.
    """
    history: List[Dict[str, Any]] = []
    base_date = datetime(2026, 8, 30)

    # Base market levels
    bdi_curve = [
        1680, 1695, 1710, 1705, 1720, 1735, 1750, 1740, 1765, 1780,
        1790, 1775, 1760, 1770, 1795, 1810, 1830, 1845, 1835, 1820,
        1810, 1795, 1805, 1825, 1850, 1865, 1880, 1870, 1855, 1840,
        1830, 1815, 1820, 1835, 1850, 1875, 1890, 1910, 1925, 1905,
        1885, 1870, 1860, 1875, 1895, 1915, 1930, 1920, 1895, 1880,
        1865, 1850, 1840, 1835, 1845, 1860, 1875, 1880, 1865, 1850,
    ]

    vlsfo_base = 585.0
    ifo380_base = 430.0
    mgo_base = 760.0

    for i in range(len(bdi_curve)):
        day_offset = len(bdi_curve) - 1 - i
        rec_date = (base_date - timedelta(days=day_offset)).strftime("%Y-%m-%d")
        close_val = float(bdi_curve[i])
        open_val = round(close_val - ((i % 5) - 2) * 4, 1)
        high_val = round(max(close_val, open_val) + (i % 4) * 5, 1)
        low_val = round(min(close_val, open_val) - (i % 3) * 6, 1)

        vlsfo_val = round(vlsfo_base + ((i % 7) - 3) * 3.5, 2)
        ifo_val = round(ifo380_base + ((i % 5) - 2) * 2.8, 2)
        mgo_val = round(mgo_base + ((i % 9) - 4) * 4.0, 2)
        hi5_val = round(vlsfo_val - ifo_val, 2)

        # 7D MA calculation
        slice_7d = bdi_curve[max(0, i - 6) : i + 1]
        ma_7d = round(sum(slice_7d) / len(slice_7d), 2)

        # 14D MA calculation
        slice_14d = bdi_curve[max(0, i - 13) : i + 1]
        ma_14d = round(sum(slice_14d) / len(slice_14d), 2)

        # 30D Volatility calculation
        slice_30d = bdi_curve[max(0, i - 29) : i + 1]
        mean_30 = sum(slice_30d) / len(slice_30d)
        var_30 = sum((x - mean_30) ** 2 for x in slice_30d) / max(1, len(slice_30d))
        vol_30d = round(var_30 ** 0.5, 2)

        history.append({
            "id": f"rec-{rec_date}",
            "date": rec_date,
            "BDI_Close": close_val,
            "BDI_Open": open_val,
            "BDI_High": high_val,
            "BDI_Low": low_val,
            "Bunker_VLSFO": vlsfo_val,
            "Bunker_MGO": mgo_val,
            "Bunker_IFO380": ifo_val,
            "Hi5_Spread": hi5_val,
            "BDI_7D_MA": ma_7d,
            "BDI_14D_MA": ma_14d,
            "BDI_30D_Vol": vol_30d,
        })

    return history


def generate_seed_scenarios() -> List[Dict[str, Any]]:
    """Seed initial representative chartering scenarios."""
    return [
        {
            "id": "scen-001",
            "created_at": "2026-08-28T10:30:00Z",
            "title": "Iron Ore Tubarao to Qingdao Capesize Run",
            "cargo_tonnes": 165000,
            "origin": "Tubarao",
            "destination": "Qingdao",
            "bdi_rate": 1850.0,
            "vlsfo_price": 585.0,
            "recommended_vessel": "Capesize",
            "optimal_landed_pmt": 27.15,
            "total_cost_usd": 4479750.0,
            "status": "APPROVED",
        },
        {
            "id": "scen-002",
            "created_at": "2026-08-29T14:15:00Z",
            "title": "Coking Coal Port Hedland to Dhamra Run",
            "cargo_tonnes": 75000,
            "origin": "Port Hedland",
            "destination": "Dhamra",
            "bdi_rate": 1840.0,
            "vlsfo_price": 580.0,
            "recommended_vessel": "Panamax",
            "optimal_landed_pmt": 15.36,
            "total_cost_usd": 1152000.0,
            "status": "UNDER_REVIEW",
        },
        {
            "id": "scen-003",
            "created_at": "2026-08-30T09:00:00Z",
            "title": "Thermal Coal Newcastle to Haldia Draft Restricted",
            "cargo_tonnes": 54000,
            "origin": "Newcastle",
            "destination": "Haldia",
            "bdi_rate": 1850.0,
            "vlsfo_price": 585.0,
            "recommended_vessel": "Supramax",
            "optimal_landed_pmt": 24.80,
            "total_cost_usd": 1339200.0,
            "status": "SIMULATED",
        },
    ]


class MaritimeDatabase:
    """
    Persistence manager for Maritime Freight Intelligence.
    Interacts with Supabase if configured; otherwise seamlessly uses in-memory seed store.
    """

    def __init__(self):
        self._supabase_client = None
        self._is_connected = False
        self._in_memory_market = generate_seed_market_history()
        self._in_memory_scenarios = generate_seed_scenarios()
        self._initialize_supabase()

    def _initialize_supabase(self):
        """Attempt connection to Supabase if credentials are present."""
        clean_url = get_clean_supabase_url()
        key = (
            os.getenv("SUPABASE_SERVICE_ROLE_KEY", "").strip()
            or os.getenv("SUPABASE_ANON_KEY", "").strip()
            or os.getenv("SUPABASE_KEY", "").strip()
        )

        if clean_url and key:
            try:
                from supabase import create_client

                self._supabase_client = create_client(clean_url, key)
                self._is_connected = True
                logger.info(f"Connected to Supabase PostgreSQL at {clean_url}")
            except Exception as e:
                logger.warning(f"Failed to initialize Supabase client: {e}. Using local store.")
                self._is_connected = False
                self._supabase_client = None
        else:
            logger.info("No SUPABASE_URL / SUPABASE_KEY provided. Operating in high-performance local store mode.")
            self._is_connected = False

    def get_status(self) -> Dict[str, Any]:
        """Return persistence layer health and connection type."""
        return {
            "is_supabase_connected": self._is_connected,
            "engine": "Supabase (PostgreSQL)" if self._is_connected else "Local Resilient Store (In-Memory/Seed)",
            "market_records_count": len(self._in_memory_market),
            "saved_scenarios_count": len(self._in_memory_scenarios),
        }

    def get_market_history(self, limit: int = 60) -> List[Dict[str, Any]]:
        """Retrieve recent market records."""
        if self._is_connected and self._supabase_client:
            try:
                response = (
                    self._supabase_client.table("market_history")
                    .select("*")
                    .order("date", desc=True)
                    .limit(limit)
                    .execute()
                )
                if response.data:
                    return list(reversed(response.data))
            except Exception as e:
                logger.error(f"Supabase market_history query failed: {e}. Falling back to local store.")

        # Fallback local store (chronological order)
        return self._in_memory_market[-limit:]

    def get_latest_market_snapshot(self) -> Dict[str, Any]:
        """Get the latest recorded day's market metrics."""
        history = self.get_market_history(limit=1)
        if history:
            return history[-1]
        return {
            "date": datetime.now().strftime("%Y-%m-%d"),
            "BDI_Close": 1850.0,
            "BDI_Open": 1845.0,
            "BDI_High": 1860.0,
            "BDI_Low": 1835.0,
            "Bunker_VLSFO": 585.0,
            "Bunker_MGO": 760.0,
            "Bunker_IFO380": 430.0,
            "Hi5_Spread": 155.0,
            "BDI_7D_MA": 1842.0,
            "BDI_14D_MA": 1838.0,
            "BDI_30D_Vol": 28.5,
        }

    def save_market_entry(self, entry: Dict[str, Any]) -> Dict[str, Any]:
        """Save a new daily market snapshot."""
        rec = dict(entry)
        if "id" not in rec:
            rec["id"] = f"rec-{rec.get('date', datetime.now().strftime('%Y-%m-%d'))}"
        if "date" not in rec:
            rec["date"] = datetime.now().strftime("%Y-%m-%d")

        if self._is_connected and self._supabase_client:
            try:
                self._supabase_client.table("market_history").upsert(rec).execute()
            except Exception as e:
                logger.error(f"Failed to upsert market record to Supabase: {e}")

        # Update in-memory
        existing_idx = next((i for i, r in enumerate(self._in_memory_market) if r["date"] == rec["date"]), None)
        if existing_idx is not None:
            self._in_memory_market[existing_idx] = rec
        else:
            self._in_memory_market.append(rec)

        return rec

    def get_saved_scenarios(self, limit: int = 20) -> List[Dict[str, Any]]:
        """Retrieve saved chartering scenarios."""
        if self._is_connected and self._supabase_client:
            try:
                response = (
                    self._supabase_client.table("charter_scenarios")
                    .select("*")
                    .order("created_at", desc=True)
                    .limit(limit)
                    .execute()
                )
                if response.data:
                    return response.data
            except Exception as e:
                logger.error(f"Supabase charter_scenarios query failed: {e}. Falling back to local store.")

        return list(reversed(self._in_memory_scenarios[-limit:]))

    def save_scenario(self, scenario: Dict[str, Any]) -> Dict[str, Any]:
        """Save or bookmark a chartering scenario."""
        rec = dict(scenario)
        if "id" not in rec:
            rec["id"] = f"scen-{uuid.uuid4().hex[:8]}"
        if "created_at" not in rec:
            rec["created_at"] = datetime.now().isoformat() + "Z"
        if "status" not in rec:
            rec["status"] = "SAVED"

        if self._is_connected and self._supabase_client:
            try:
                self._supabase_client.table("charter_scenarios").insert(rec).execute()
            except Exception as e:
                logger.error(f"Failed to insert scenario into Supabase: {e}")

        self._in_memory_scenarios.append(rec)
        return rec


# Global Database instance
db_instance = MaritimeDatabase()
