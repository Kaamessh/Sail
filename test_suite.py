"""
Comprehensive Automated Test Suite for Maritime Freight Intelligence System.
Tests:
1. FreightPredictor singleton, quantile monotonicity (P10 <= P50 <= P90), derived features.
2. CharterOptimizer draft restrictions, landed $/Tonne calculation, MILP/capacity math.
3. FastAPI API endpoints: /api/health, /api/predict, /api/optimize, /api/stress-test, /api/history.
"""

import asyncio
import unittest
import httpx

from backend.main import app
from backend.optimizer import GLOBAL_PORTS, CharterOptimizer
from models.predictor import FreightPredictor


class TestMaritimeFreightSystem(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.predictor = FreightPredictor.get_instance()
        cls.transport = httpx.ASGITransport(app=app)

    def _api_get(self, path: str):
        async def _call():
            async with httpx.AsyncClient(transport=self.transport, base_url="http://testserver") as client:
                return await client.get(path)
        return asyncio.run(_call())

    def _api_post(self, path: str, json_data: dict):
        async def _call():
            async with httpx.AsyncClient(transport=self.transport, base_url="http://testserver") as client:
                return await client.post(path, json=json_data)
        return asyncio.run(_call())

    def test_01_predictor_initialization_and_metadata(self):
        """Verify predictor loaded models for 7D, 14D, 30D and scalers."""
        self.assertTrue(self.predictor.is_ready)
        self.assertIn(7, self.predictor._models)
        self.assertIn(14, self.predictor._models)
        self.assertIn(30, self.predictor._models)
        self.assertIn("scaler_7d", self.predictor._scalers)
        self.assertIn("scaler_14d", self.predictor._scalers)
        self.assertIn("scaler_30d", self.predictor._scalers)

    def test_02_predictor_quantile_forecasts_and_monotonicity(self):
        """Verify predictions adhere to P10 <= P50 <= P90 for all horizons."""
        sample_input = {
            "BDI_Close": 1850.0,
            "BDI_Open": 1840.0,
            "BDI_High": 1865.0,
            "BDI_Low": 1830.0,
            "Bunker_VLSFO": 585.0,
            "Bunker_MGO": 760.0,
            "Bunker_IFO380": 430.0,
        }
        res = self.predictor.predict(sample_input)
        self.assertIn("forecasts", res)
        self.assertIn("7D", res["forecasts"])
        self.assertIn("14D", res["forecasts"])
        self.assertIn("30D", res["forecasts"])

        for h in ["7D", "14D", "30D"]:
            f = res["forecasts"][h]
            p10, p50, p90 = f["p10"], f["p50"], f["p90"]
            self.assertLessEqual(p10, p50, f"P10 ({p10}) should be <= P50 ({p50}) for horizon {h}")
            self.assertLessEqual(p50, p90, f"P50 ({p50}) should be <= P90 ({p90}) for horizon {h}")
            self.assertGreater(p10, 0)
            self.assertGreater(p50, 0)
            self.assertGreater(p90, 0)

        self.assertIn("trend_analysis", res)
        self.assertEqual(res["trend_analysis"]["hi5_spread"], 155.0)

    def test_03_derived_features_computation(self):
        """Verify automated calculation of Hi5 spread, moving averages, and volatility."""
        derived = self.predictor.compute_derived_features({
            "BDI_Close": 2000.0,
            "Bunker_VLSFO": 600.0,
            "Bunker_IFO380": 450.0,
        })
        self.assertEqual(derived["Hi5_Spread"], 150.0)
        self.assertEqual(derived["BDI_Close"], 2000.0)
        self.assertIn("BDI_7D_MA", derived)
        self.assertIn("BDI_14D_MA", derived)
        self.assertIn("BDI_30D_Vol", derived)

    def test_04_optimizer_draft_constraints_haldia(self):
        """Haldia 8.5m draft must restrict Capesize (18.2m) & Panamax (14.2m)."""
        res = CharterOptimizer.optimize_charter(
            cargo_tonnes=60000,
            origin="Newcastle",
            destination="Haldia",
        )
        # Capesize and Panamax should fail draft test
        options = {v["vessel_class"]: v for v in res["vessel_options"]}
        self.assertFalse(options["Capesize"]["is_feasible"])
        self.assertFalse(options["Panamax"]["is_feasible"])
        self.assertEqual(options["Capesize"]["port_draft_limit_m"], 8.5)
        self.assertEqual(options["Panamax"]["port_draft_limit_m"], 8.5)

    def test_05_optimizer_deepwater_capesize_selection(self):
        """Tubarao to Qingdao (both > 21m draft) with 165k cargo must choose Capesize as lowest landed $/Tonne."""
        res = CharterOptimizer.optimize_charter(
            cargo_tonnes=165000,
            origin="Tubarao",
            destination="Qingdao",
            bdi_forecast=1850.0,
            vlsfo_price=585.0,
        )
        self.assertEqual(res["recommended_vessel"], "Capesize")
        self.assertEqual(res["optimization_status"], "OPTIMAL")
        self.assertTrue(res["vessel_options"][0]["is_feasible"])
        self.assertLess(res["optimal_landed_cost_pmt"], 35.0)

    def test_06_api_health_endpoint(self):
        """Test GET /api/health."""
        response = self._api_get("/api/health")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["status"], "healthy")
        self.assertTrue(data["model_artifacts_loaded"])

    def test_07_api_predict_endpoint(self):
        """Test POST /api/predict."""
        payload = {
            "BDI_Close": 1920.0,
            "Bunker_VLSFO": 610.0,
            "Bunker_IFO380": 460.0,
        }
        response = self._api_post("/api/predict", json_data=payload)
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("forecasts", data)
        self.assertIn("7D", data["forecasts"])
        self.assertIn("14D", data["forecasts"])
        self.assertIn("30D", data["forecasts"])

    def test_08_api_optimize_endpoint(self):
        """Test POST /api/optimize."""
        payload = {
            "cargo_tonnes": 80000,
            "origin_port": "Port Hedland",
            "destination_port": "Dhamra",
            "bdi_forecast_override": 1850.0,
        }
        response = self._api_post("/api/optimize", json_data=payload)
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("recommended_vessel", data)
        self.assertIn("optimal_landed_cost_pmt", data)
        self.assertGreater(len(data["vessel_options"]), 0)

    def test_09_api_stress_test_endpoint(self):
        """Test POST /api/stress-test."""
        payload = {
            "cargo_tonnes": 75000,
            "origin_port": "Port Hedland",
            "destination_port": "Qingdao",
            "bunker_spikes": [0.0, 10.0, 25.0],
            "bdi_shifts": [0.0, 20.0],
        }
        response = self._api_post("/api/stress-test", json_data=payload)
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("stress_matrix", data)
        self.assertEqual(len(data["stress_matrix"]), 6)

    def test_10_api_history_and_save_scenario(self):
        """Test GET & POST /api/history."""
        # Get history
        get_res = self._api_get("/api/history")
        self.assertEqual(get_res.status_code, 200)
        history_data = get_res.json()
        self.assertGreater(history_data["count"], 0)

        # Save scenario
        new_scenario = {
            "title": "Automated Test Scenario Run",
            "cargo_tonnes": 70000,
            "origin": "Santos",
            "destination": "Qingdao",
            "bdi_rate": 1850.0,
            "vlsfo_price": 590.0,
            "recommended_vessel": "Panamax",
            "optimal_landed_pmt": 32.50,
            "total_cost_usd": 2275000.0,
        }
        post_res = self._api_post("/api/history", json_data=new_scenario)
        self.assertEqual(post_res.status_code, 200)
        self.assertEqual(post_res.json()["status"], "saved")


if __name__ == "__main__":
    unittest.main(verbosity=2)
