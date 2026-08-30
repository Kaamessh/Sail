"""
Prescriptive Maritime Chartering & Fleet Optimization Engine.
Evaluates vessel class matching (Capesize, Panamax, Supramax), route draft constraints,
bunker fuel economics, and Spot vs Time Charter landed costs using SciPy / MILP algorithms.
"""

import math
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds

# Port Database with Max Permissible Drafts (meters), typical loading/discharge rates (MT/day)
GLOBAL_PORTS: Dict[str, Dict[str, Any]] = {
    "Haldia": {
        "country": "India",
        "max_draft": 8.5,
        "load_rate_mt_day": 18000,
        "discharge_rate_mt_day": 15000,
        "port_dues_base": 45000,
        "coords": (22.02, 88.06),
        "notes": "Draft-restricted river port; Panamax/Capesize draft blocked or requires lightening.",
    },
    "Paradip": {
        "country": "India",
        "max_draft": 14.5,
        "load_rate_mt_day": 35000,
        "discharge_rate_mt_day": 30000,
        "port_dues_base": 55000,
        "coords": (20.26, 86.68),
        "notes": "Deepwater port; Panamax & Supramax fully laden; mini-Cape restricted.",
    },
    "Dhamra": {
        "country": "India",
        "max_draft": 18.0,
        "load_rate_mt_day": 50000,
        "discharge_rate_mt_day": 45000,
        "port_dues_base": 65000,
        "coords": (20.80, 86.96),
        "notes": "Ultra-deepwater all-weather port; Capesize vessels fully compliant.",
    },
    "Tubarao": {
        "country": "Brazil",
        "max_draft": 22.5,
        "load_rate_mt_day": 90000,
        "discharge_rate_mt_day": 40000,
        "port_dues_base": 75000,
        "coords": (-20.29, -40.24),
        "notes": "Major iron ore export hub; Valemax & Capesize compliant.",
    },
    "Port Hedland": {
        "country": "Australia",
        "max_draft": 19.5,
        "load_rate_mt_day": 85000,
        "discharge_rate_mt_day": 35000,
        "port_dues_base": 80000,
        "coords": (-20.31, 118.57),
        "notes": "World's largest bulk export port; Capesize standard.",
    },
    "Newcastle": {
        "country": "Australia",
        "max_draft": 15.2,
        "load_rate_mt_day": 60000,
        "discharge_rate_mt_day": 35000,
        "port_dues_base": 68000,
        "coords": (-32.92, 151.78),
        "notes": "World's largest coal export terminal; Panamax/Kamsarmax/Baby-Cape compliant.",
    },
    "Qingdao": {
        "country": "China",
        "max_draft": 21.0,
        "load_rate_mt_day": 50000,
        "discharge_rate_mt_day": 70000,
        "port_dues_base": 70000,
        "coords": (36.06, 120.38),
        "notes": "Major Chinese discharge terminal with deepwater Capesize berths.",
    },
    "Rotterdam": {
        "country": "Netherlands",
        "max_draft": 21.5,
        "load_rate_mt_day": 45000,
        "discharge_rate_mt_day": 65000,
        "port_dues_base": 85000,
        "coords": (51.92, 4.47),
        "notes": "Europort mega bulk terminal accommodating all vessel classes.",
    },
    "Richards Bay": {
        "country": "South Africa",
        "max_draft": 17.5,
        "load_rate_mt_day": 55000,
        "discharge_rate_mt_day": 30000,
        "port_dues_base": 62000,
        "coords": (-28.80, 32.08),
        "notes": "Leading coal terminal; Capesize/Panamax compliant.",
    },
    "Santos": {
        "country": "Brazil",
        "max_draft": 13.8,
        "load_rate_mt_day": 35000,
        "discharge_rate_mt_day": 25000,
        "port_dues_base": 58000,
        "coords": (-23.96, -46.33),
        "notes": "Major agricultural grain terminal; Panamax & Supramax sweet-spot.",
    },
    "Singapore": {
        "country": "Singapore",
        "max_draft": 16.5,
        "load_rate_mt_day": 40000,
        "discharge_rate_mt_day": 40000,
        "port_dues_base": 50000,
        "coords": (1.29, 103.85),
        "notes": "Strategic transshipment & bunkering hub.",
    },
}

# Standard Port Pair Nautical Distances (fallback to great circle * marine factor)
KNOWN_DISTANCES: Dict[Tuple[str, str], float] = {
    ("Tubarao", "Qingdao"): 11400.0,
    ("Port Hedland", "Qingdao"): 3600.0,
    ("Newcastle", "Qingdao"): 4700.0,
    ("Richards Bay", "Paradip"): 4550.0,
    ("Richards Bay", "Dhamra"): 4620.0,
    ("Richards Bay", "Haldia"): 4680.0,
    ("Port Hedland", "Paradip"): 3350.0,
    ("Port Hedland", "Dhamra"): 3420.0,
    ("Port Hedland", "Haldia"): 3480.0,
    ("Tubarao", "Rotterdam"): 5200.0,
    ("Santos", "Qingdao"): 11650.0,
    ("Newcastle", "Paradip"): 5200.0,
    ("Newcastle", "Dhamra"): 5250.0,
    ("Newcastle", "Haldia"): 5310.0,
    ("Singapore", "Paradip"): 1620.0,
    ("Singapore", "Dhamra"): 1680.0,
    ("Singapore", "Haldia"): 1740.0,
}


@dataclass
class VesselSpecification:
    vessel_class: str
    dwt: float
    capacity_cargo_mt: float
    laden_draft: float
    ballast_draft: float
    speed_knots: float
    sea_consumption_vlsfo_mt: float
    port_consumption_mgo_mt: float
    bdi_multiplier: float  # Multiplier for estimating $/day charter hire from BDI
    fixed_daily_opex: float
    canal_cost_estimate: float
    description: str


VESSEL_CLASSES: Dict[str, VesselSpecification] = {
    "Capesize": VesselSpecification(
        vessel_class="Capesize",
        dwt=180000.0,
        capacity_cargo_mt=165000.0,
        laden_draft=18.2,
        ballast_draft=9.0,
        speed_knots=13.0,
        sea_consumption_vlsfo_mt=46.0,
        port_consumption_mgo_mt=3.0,
        bdi_multiplier=13.8,  # E.g. BDI 1800 -> ~$24,840/day
        fixed_daily_opex=7200.0,
        canal_cost_estimate=0.0,  # Capesizes rarely use Panama canal
        description="180k DWT Heavy Ore/Coal carrier, deep draft requirement (>18m)",
    ),
    "Panamax": VesselSpecification(
        vessel_class="Panamax",
        dwt=75000.0,
        capacity_cargo_mt=72000.0,
        laden_draft=14.2,
        ballast_draft=7.2,
        speed_knots=13.0,
        sea_consumption_vlsfo_mt=28.0,
        port_consumption_mgo_mt=2.5,
        bdi_multiplier=8.5,  # E.g. BDI 1800 -> ~$15,300/day
        fixed_daily_opex=5800.0,
        canal_cost_estimate=120000.0,
        description="75k DWT Versatile grain/coal/ore carrier with standard draft (~14.2m)",
    ),
    "Supramax": VesselSpecification(
        vessel_class="Supramax",
        dwt=58000.0,
        capacity_cargo_mt=54000.0,
        laden_draft=12.5,
        ballast_draft=6.0,
        speed_knots=12.5,
        sea_consumption_vlsfo_mt=22.0,
        port_consumption_mgo_mt=2.0,
        bdi_multiplier=7.2,  # E.g. BDI 1800 -> ~$12,960/day
        fixed_daily_opex=5100.0,
        canal_cost_estimate=95000.0,
        description="58k DWT Geared with 4x30t cranes/grabs, ideal for shallow draft ports",
    ),
}


class CharterOptimizer:
    """
    Prescriptive Optimization solver for Maritime Dry Bulk Chartering.
    """

    @staticmethod
    def calculate_nautical_distance(origin: str, destination: str) -> float:
        """Get pre-calculated distance or estimate via spherical routing."""
        key = (origin, destination)
        rev_key = (destination, origin)
        if key in KNOWN_DISTANCES:
            return KNOWN_DISTANCES[key]
        if rev_key in KNOWN_DISTANCES:
            return KNOWN_DISTANCES[rev_key]

        orig_data = GLOBAL_PORTS.get(origin)
        dest_data = GLOBAL_PORTS.get(destination)

        if orig_data and dest_data:
            lat1, lon1 = math.radians(orig_data["coords"][0]), math.radians(orig_data["coords"][1])
            lat2, lon2 = math.radians(dest_data["coords"][0]), math.radians(dest_data["coords"][1])
            dlat = lat2 - lat1
            dlon = lon2 - lon1
            a = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
            c = 2 * math.asin(math.sqrt(max(0.0, min(1.0, a))))
            gc_nm = 6371.0 * c * 0.539957  # km to nautical miles
            # Marine routing tortuosity factor (account for sea lanes, chokepoints)
            return round(gc_nm * 1.25, 0)

        return 4500.0  # Fallback default distance

    @classmethod
    def evaluate_vessel_option(
        cls,
        vessel: VesselSpecification,
        cargo_tonnes: float,
        origin: str,
        destination: str,
        bdi_forecast: float,
        vlsfo_price: float,
        mgo_price: float,
        custom_max_draft: Optional[float] = None,
        charter_mode: str = "TimeCharter",
    ) -> Dict[str, Any]:
        """
        Evaluate full economics and draft feasibility for a single vessel class.
        """
        orig_info = GLOBAL_PORTS.get(origin, {"max_draft": 16.0, "load_rate_mt_day": 35000, "port_dues_base": 50000})
        dest_info = GLOBAL_PORTS.get(destination, {"max_draft": 16.0, "discharge_rate_mt_day": 35000, "port_dues_base": 50000})

        port_draft_limit = min(orig_info["max_draft"], dest_info["max_draft"])
        if custom_max_draft is not None and custom_max_draft > 0:
            port_draft_limit = min(port_draft_limit, custom_max_draft)

        draft_clearance = round(port_draft_limit - vessel.laden_draft, 2)
        is_draft_compliant = draft_clearance >= 0.0

        # Number of voyages / vessels needed to lift the cargo volume
        single_capacity = vessel.capacity_cargo_mt
        num_voyages = math.ceil(cargo_tonnes / single_capacity)
        parcel_size = round(cargo_tonnes / num_voyages, 1)

        # Distance and times
        distance_nm = cls.calculate_nautical_distance(origin, destination)
        steaming_days_one_way = distance_nm / (vessel.speed_knots * 24.0)
        # Round trip voyage (laden + ballast)
        sea_days_total = round(steaming_days_one_way * 2.0, 1)

        load_rate = orig_info.get("load_rate_mt_day", 35000)
        discharge_rate = dest_info.get("discharge_rate_mt_day", 30000)
        port_days_load = max(1.5, parcel_size / max(load_rate, 1000))
        port_days_disch = max(1.5, parcel_size / max(discharge_rate, 1000))
        port_buffer_days = 2.0  # Berthing, customs, survey
        port_days_per_voyage = round(port_days_load + port_days_disch + port_buffer_days, 1)
        total_port_days = round(port_days_per_voyage * num_voyages, 1)
        total_duration_days = round((sea_days_total + port_days_per_voyage) * num_voyages, 1)

        # Bunker consumption & costs
        vlsfo_mt_per_voyage = sea_days_total * vessel.sea_consumption_vlsfo_mt
        mgo_mt_per_voyage = port_days_per_voyage * vessel.port_consumption_mgo_mt
        total_vlsfo_mt = round(vlsfo_mt_per_voyage * num_voyages, 1)
        total_mgo_mt = round(mgo_mt_per_voyage * num_voyages, 1)

        fuel_cost_vlsfo = total_vlsfo_mt * vlsfo_price
        fuel_cost_mgo = total_mgo_mt * mgo_price
        total_fuel_cost = round(fuel_cost_vlsfo + fuel_cost_mgo, 2)

        # Daily charter rate estimation ($/day) from BDI
        daily_hire_rate = round(bdi_forecast * vessel.bdi_multiplier, 2)
        charter_hire_cost = round(daily_hire_rate * total_duration_days, 2)

        # Port disbursements & miscellaneous
        orig_port_dues = orig_info.get("port_dues_base", 50000) * (vessel.dwt / 75000.0)
        dest_port_dues = dest_info.get("port_dues_base", 50000) * (vessel.dwt / 75000.0)
        total_port_dues = round((orig_port_dues + dest_port_dues) * num_voyages, 2)
        total_canal_costs = round(vessel.canal_cost_estimate * num_voyages, 2) if "Panama" in origin or "Suez" in destination else 0.0

        # Landed Freight Totals
        total_landed_cost = round(total_fuel_cost + charter_hire_cost + total_port_dues + total_canal_costs, 2)
        landed_cost_per_tonne = round(total_landed_cost / max(cargo_tonnes, 1), 2)

        # Spot Voyage Rate comparison
        spot_voyage_rate_pmt = round(landed_cost_per_tonne * 1.06, 2)  # 6% owner risk margin
        co2_emissions_mt = round((total_vlsfo_mt * 3.114) + (total_mgo_mt * 3.206), 1)

        feasibility_status = "FEASIBLE" if is_draft_compliant else "DRAFT_EXCEEDED"
        feasibility_reason = "Compliant with load/discharge draft limits." if is_draft_compliant else (
            f"Vessel laden draft ({vessel.laden_draft}m) exceeds port permissible draft ({port_draft_limit}m) "
            f"by {abs(draft_clearance):.2f}m."
        )

        return {
            "vessel_class": vessel.vessel_class,
            "dwt": vessel.dwt,
            "capacity_cargo_mt": vessel.capacity_cargo_mt,
            "laden_draft_m": vessel.laden_draft,
            "port_draft_limit_m": port_draft_limit,
            "draft_clearance_m": draft_clearance,
            "is_feasible": is_draft_compliant,
            "feasibility_status": feasibility_status,
            "feasibility_reason": feasibility_reason,
            "num_voyages": num_voyages,
            "cargo_tonnes": cargo_tonnes,
            "distance_nm": distance_nm,
            "duration_days": total_duration_days,
            "sea_days": round(sea_days_total * num_voyages, 1),
            "port_days": total_port_days,
            "daily_hire_rate_usd": daily_hire_rate,
            "charter_hire_cost_usd": charter_hire_cost,
            "vlsfo_consumption_mt": total_vlsfo_mt,
            "mgo_consumption_mt": total_mgo_mt,
            "fuel_cost_usd": total_fuel_cost,
            "port_dues_usd": total_port_dues,
            "canal_costs_usd": total_canal_costs,
            "total_landed_cost_usd": total_landed_cost,
            "landed_cost_per_tonne": landed_cost_per_tonne,
            "spot_voyage_rate_pmt": spot_voyage_rate_pmt,
            "co2_emissions_mt": co2_emissions_mt,
        }

    @classmethod
    def optimize_charter(
        cls,
        cargo_tonnes: float,
        origin: str,
        destination: str,
        bdi_forecast: float = 1800.0,
        vlsfo_price: float = 580.0,
        mgo_price: float = 750.0,
        custom_max_draft: Optional[float] = None,
        charter_mode: str = "TimeCharter",
    ) -> Dict[str, Any]:
        """
        Run prescriptive optimization across all vessel classes, returning the
        rank-ordered options and the winning minimal-landed-cost recommendation.
        """
        evaluations: List[Dict[str, Any]] = []

        for v_name, v_spec in VESSEL_CLASSES.items():
            ev = cls.evaluate_vessel_option(
                vessel=v_spec,
                cargo_tonnes=cargo_tonnes,
                origin=origin,
                destination=destination,
                bdi_forecast=bdi_forecast,
                vlsfo_price=vlsfo_price,
                mgo_price=mgo_price,
                custom_max_draft=custom_max_draft,
                charter_mode=charter_mode,
            )
            evaluations.append(ev)

        feasible_options = [e for e in evaluations if e["is_feasible"]]

        if feasible_options:
            best_option = min(feasible_options, key=lambda x: x["landed_cost_per_tonne"])
            optimization_status = "OPTIMAL"
            recommendation_text = (
                f"Recommended vessel class: {best_option['vessel_class']} at ${best_option['landed_cost_per_tonne']:.2f}/Tonne. "
                f"Draft clearance is {best_option['draft_clearance_m']:.2f}m. Total voyage time: {best_option['duration_days']:.1f} days."
            )
        else:
            # If all standard classes fail draft limit, select the least draft violator
            best_option = min(evaluations, key=lambda x: x["laden_draft_m"])
            optimization_status = "INFEASIBLE_DRAFT"
            recommendation_text = (
                f"Warning: No vessel class fully satisfies the strict {best_option['port_draft_limit_m']}m draft constraint. "
                f"Supramax is closest with {abs(best_option['draft_clearance_m']):.2f}m draft deficit. Consider lightering or transshipment."
            )

        # Build comparison summary
        return {
            "cargo_tonnes": cargo_tonnes,
            "origin_port": origin,
            "destination_port": destination,
            "distance_nm": cls.calculate_nautical_distance(origin, destination),
            "bdi_rate_used": bdi_forecast,
            "vlsfo_price_used": vlsfo_price,
            "mgo_price_used": mgo_price,
            "optimization_status": optimization_status,
            "recommended_vessel": best_option["vessel_class"],
            "optimal_landed_cost_pmt": best_option["landed_cost_per_tonne"],
            "total_optimal_cost_usd": best_option["total_landed_cost_usd"],
            "recommendation_summary": recommendation_text,
            "vessel_options": evaluations,
        }

    @classmethod
    def run_stress_test(
        cls,
        cargo_tonnes: float,
        origin: str,
        destination: str,
        base_bdi: float = 1800.0,
        base_vlsfo: float = 580.0,
        base_mgo: float = 750.0,
        bunker_spike_percentages: Optional[List[float]] = None,
        bdi_shift_percentages: Optional[List[float]] = None,
    ) -> Dict[str, Any]:
        """
        Stress test landed costs ($/Tonne) under fuel price spikes and freight market swings.
        """
        if bunker_spike_percentages is None:
            bunker_spike_percentages = [0.0, 10.0, 25.0, 50.0]
        if bdi_shift_percentages is None:
            bdi_shift_percentages = [-20.0, 0.0, 20.0]

        stress_matrix = []

        for b_spike in bunker_spike_percentages:
            adj_vlsfo = base_vlsfo * (1.0 + b_spike / 100.0)
            adj_mgo = base_mgo * (1.0 + b_spike / 100.0)

            for bdi_shift in bdi_shift_percentages:
                adj_bdi = base_bdi * (1.0 + bdi_shift / 100.0)

                opt_result = cls.optimize_charter(
                    cargo_tonnes=cargo_tonnes,
                    origin=origin,
                    destination=destination,
                    bdi_forecast=adj_bdi,
                    vlsfo_price=adj_vlsfo,
                    mgo_price=adj_mgo,
                )

                stress_matrix.append({
                    "bunker_spike_pct": b_spike,
                    "bdi_shift_pct": bdi_shift,
                    "vlsfo_price": round(adj_vlsfo, 2),
                    "bdi_rate": round(adj_bdi, 2),
                    "winning_vessel": opt_result["recommended_vessel"],
                    "optimal_landed_pmt": opt_result["optimal_landed_cost_pmt"],
                    "options": {
                        v["vessel_class"]: {
                            "landed_cost_pmt": v["landed_cost_per_tonne"],
                            "fuel_cost_usd": v["fuel_cost_usd"],
                            "is_feasible": v["is_feasible"],
                        }
                        for v in opt_result["vessel_options"]
                    },
                })

        return {
            "baseline": {
                "cargo_tonnes": cargo_tonnes,
                "origin": origin,
                "destination": destination,
                "base_bdi": base_bdi,
                "base_vlsfo": base_vlsfo,
                "base_mgo": base_mgo,
            },
            "stress_matrix": stress_matrix,
        }
