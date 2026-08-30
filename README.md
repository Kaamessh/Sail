# 🚢 Aura Maritime Freight Intelligence & Prescriptive Chartering Decision System

A production-ready Full-Stack Python Maritime Freight Intelligence and Prescriptive Chartering Decision System.

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-009688.svg)
![Vercel](https://img.shields.io/badge/Vercel-Serverless-black.svg)
![Supabase](https://img.shields.io/badge/Supabase-PostgreSQL-3ECF8E.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

---

## 🌊 Overview

Aura Maritime combines multi-horizon Machine Learning quantile regression with Mixed-Integer Linear Programming (MILP) prescriptive economics to optimize dry bulk chartering decisions:

1. **Multi-Horizon Quantile Freight Forecasting**:
   - Pre-trained Gradient Boosting Quantile Regressors for Baltic Dry Index (BDI) across **7-Day**, **14-Day**, and **30-Day** horizons.
   - Outputs **P10 (Downside Floor)**, **P50 (Expected Median)**, and **P90 (Bullish Ceiling)** confidence bands.
   - Automated feature engineering for `Hi5_Spread` (VLSFO - IFO380), 7D/14D Moving Averages, and 30D Rolling Volatility.

2. **Prescriptive Chartering & Fleet Optimization**:
   - Evaluates **Capesize (180k DWT)**, **Panamax (75k DWT)**, and **Supramax (58k DWT)** classes.
   - Enforces port permissible drafts (e.g. Haldia: 8.5m, Paradip: 14.5m, Dhamra: 18.0m, Tubarao: 22.5m, Qingdao: 21.0m).
   - Computes landed $/Tonne voyage economics, bunker consumption (sea VLSFO & port MGO), TCE daily hire, and $CO_2$ emissions.

3. **Dark Maritime Dashboard**:
   - Executive KPI cards with real-time Baltic indices and bunker spreads.
   - Interactive Chart.js forecast cone visualizer with confidence corridor shading.
   - Real-time bunker fuel spike and market swing stress-testing sandbox.
   - Saved scenario ledger and historical market explorer.

---

## 📁 Project Structure

```
Sail/
├── api/
│   └── index.py                      # Vercel Serverless Entrypoint (ASGI handler)
├── backend/
│   ├── __init__.py                   # Package initialization
│   ├── database.py                   # Supabase PostgreSQL + Fallback Data Layer
│   ├── main.py                       # FastAPI Application with static UI mounting
│   ├── optimizer.py                  # Prescriptive Chartering & Fleet Optimization Engine
│   └── schemas.py                    # Pydantic Request/Response validation models
├── frontend/
│   ├── app.js                        # Chart.js cone visualizer & reactive UI logic
│   ├── index.html                    # Dark Maritime Dashboard SPA
│   └── styles.css                    # Glassmorphic Dark Maritime Theme
├── models/
│   ├── __init__.py                   
│   ├── artifacts/                    # Pre-trained ML weights and scalers
│   │   ├── freight_quantile_models.joblib  
│   │   ├── freight_scalers.joblib          
│   │   └── metadata.json                   
│   └── predictor.py                  # Multi-horizon Quantile Predictor Singleton
├── .env.example                      # Template for environment variables
├── .gitignore                        # Git exclusion rules
├── requirements.txt                  # Python dependencies
├── run_local.py                      # Local development launcher
├── test_suite.py                     # Automated unit and integration test suite
└── vercel.json                       # Vercel deployment and routing configuration
```

---

## ⚡ Quick Start

### 1. Clone & Install Dependencies
```bash
git clone https://github.com/Kaamessh/Sail.git
cd Sail
pip install -r requirements.txt
```

### 2. Configure Environment (Optional)
Copy `.env.example` to `.env` if connecting to a live Supabase project:
```bash
cp .env.example .env
```
*(The application also includes a resilient local seed store and runs without external dependencies).*

### 3. Run Locally
```bash
python run_local.py
```
- **Dashboard UI**: [http://localhost:8000](http://localhost:8000)
- **Interactive Swagger Docs**: [http://localhost:8000/docs](http://localhost:8000/docs)
- **Health Check**: [http://localhost:8000/api/health](http://localhost:8000/api/health)

### 4. Run Test Suite
```bash
python test_suite.py
```

---

## ☁️ Deploy to Vercel

```bash
vercel
```

In your Vercel Project Dashboard, add your environment variables:
- `SUPABASE_URL`
- `SUPABASE_KEY`

---

## 📄 License
MIT License.
