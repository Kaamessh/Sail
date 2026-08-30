/**
 * Maritime Freight Intelligence & Prescriptive Chartering Decision Dashboard
 * Client Application Logic - 100% Genuine ML Pipeline
 */

// Global State
const state = {
  currentMarket: null,
  history: [],
  forecasts: null,
  selectedHorizon: 'all',
  currentOptimization: null,
  savedScenarios: [],
  chartInstance: null,
  baselineOpt: null
};

// Robust API fetch helper resolving both /api and root endpoints
async function apiFetch(endpoint, options = {}) {
  try {
    const res = await fetch(`/api${endpoint}`, options);
    if (res.ok) return res;
    if (res.status === 404) {
      const resFallback = await fetch(endpoint, options);
      if (resFallback.ok) return resFallback;
    }
    return res;
  } catch (err) {
    return await fetch(endpoint, options);
  }
}

// Self-executing initialization that handles already-loaded or async DOM states
if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', initApp);
} else {
  initApp();
}

async function initApp() {
  console.log("Initializing Aura Maritime Dashboard...");
  setupEventListeners();
  await checkSystemHealth();
  await loadMarketData();
  await triggerDefaultOptimization();
}

function setupEventListeners() {
  // Horizon Tabs
  const tabs = document.querySelectorAll('#horizon-tabs .tab-btn');
  tabs.forEach(tab => {
    tab.addEventListener('click', (e) => {
      tabs.forEach(t => t.classList.remove('active'));
      e.target.classList.add('active');
      state.selectedHorizon = e.target.dataset.horizon;
      renderForecastChart();
    });
  });

  // Charter Form Submit & Real-Time Auto-Evaluation on Input Changes
  const charterForm = document.getElementById('charter-form');
  if (charterForm) {
    charterForm.addEventListener('submit', async (e) => {
      e.preventDefault();
      await runCharterOptimization();
    });
  }

  // Reactive Instant Calculation Listeners
  const cargoInput = document.getElementById('input-cargo');
  const modeSelect = document.getElementById('select-mode');
  const originSelect = document.getElementById('select-origin');
  const destSelect = document.getElementById('select-destination');
  const draftInput = document.getElementById('input-draft');

  if (cargoInput) {
    cargoInput.addEventListener('input', () => { runCharterOptimization(); });
  }
  if (modeSelect) {
    modeSelect.addEventListener('change', () => { runCharterOptimization(); });
  }
  if (originSelect && destSelect) {
    originSelect.addEventListener('change', () => {
      checkDraftAlerts();
      runCharterOptimization();
    });
    destSelect.addEventListener('change', () => {
      checkDraftAlerts();
      runCharterOptimization();
    });
  }
  if (draftInput) {
    draftInput.addEventListener('input', () => { runCharterOptimization(); });
  }

  // Refresh Market Data Button
  const btnRefresh = document.getElementById('btn-refresh-market');
  if (btnRefresh) {
    btnRefresh.addEventListener('click', async () => {
      btnRefresh.innerHTML = '<i class="fa-solid fa-spinner fa-spin"></i> Syncing...';
      await loadMarketData();
      await runCharterOptimization();
      btnRefresh.innerHTML = '<i class="fa-solid fa-rotate"></i> Sync Market';
    });
  }

  // Save Scenario Button
  const btnSave = document.getElementById('btn-save-scenario');
  if (btnSave) {
    btnSave.addEventListener('click', async () => {
      await saveCurrentScenario();
    });
  }

  // Stress-testing Sandbox Sliders
  const sliderBunker = document.getElementById('slider-bunker');
  const sliderBdi = document.getElementById('slider-bdi');

  if (sliderBunker && sliderBdi) {
    sliderBunker.addEventListener('input', updateStressTestResults);
    sliderBdi.addEventListener('input', updateStressTestResults);
  }
}

function checkDraftAlerts() {
  const origin = document.getElementById('select-origin').value;
  const dest = document.getElementById('select-destination').value;
  const draftInput = document.getElementById('input-draft');

  if (origin === 'Haldia' || dest === 'Haldia') {
    draftInput.placeholder = '8.5m (Haldia Restricted)';
  } else {
    draftInput.placeholder = 'Auto (From Port)';
  }
}

async function checkSystemHealth() {
  try {
    const res = await apiFetch('/health');
    const data = await res.json();
    const statusText = document.getElementById('system-status-text');
    const badge = document.getElementById('system-status-badge');

    if (data.status === 'healthy' && data.model_artifacts_loaded) {
      statusText.innerText = `ML Models Online (7D, 14D, 30D)`;
      badge.style.borderColor = 'rgba(16, 185, 129, 0.4)';
    } else {
      statusText.innerText = 'ML Models Degraded / Offline';
      badge.style.borderColor = 'rgba(255, 77, 109, 0.4)';
      console.error('CRITICAL: Health check returned degraded model status:', data);
    }
  } catch (err) {
    console.error('CRITICAL: Health check failed to connect:', err);
  }
}

async function loadMarketData() {
  try {
    const res = await apiFetch('/history?limit=45');
    if (!res.ok) {
      throw new Error(`Failed to fetch market history: HTTP ${res.status}`);
    }
    const data = await res.json();
    state.history = data.history || [];
    state.currentMarket = data.latest_snapshot;
    state.savedScenarios = data.saved_scenarios || [];

    if (!state.currentMarket) {
      throw new Error('API returned empty latest_snapshot.');
    }

    updateExecutiveKPIs(state.currentMarket);
    renderSavedScenarios(state.savedScenarios);

    // Run ML prediction with real current market values
    await fetchForecastCones(state.currentMarket);
  } catch (err) {
    console.error('FATAL: loadMarketData failed:', err);
  }
}

function updateExecutiveKPIs(market) {
  if (!market) return;

  const bdiClose = Number(market.BDI_Close);
  const bdi7dMa = Number(market.BDI_7D_MA);
  const bdiVol = Number(market.BDI_30D_Vol);
  const vlsfo = Number(market.Bunker_VLSFO);
  const mgo = Number(market.Bunker_MGO);
  const ifo = Number(market.Bunker_IFO380);
  const hi5 = Number(market.Hi5_Spread);

  if (!isNaN(bdiClose)) {
    document.getElementById('kpi-bdi-val').innerText = bdiClose.toLocaleString();
  }
  if (!isNaN(bdi7dMa)) {
    document.getElementById('kpi-bdi-7dma').innerText = Math.round(bdi7dMa).toLocaleString();
  }
  if (!isNaN(bdiVol)) {
    document.getElementById('kpi-bdi-vol').innerText = bdiVol.toFixed(1);
  }

  if (!isNaN(vlsfo)) {
    document.getElementById('kpi-vlsfo-val').innerText = `$${vlsfo.toFixed(2)}`;
  }
  if (!isNaN(mgo)) {
    document.getElementById('kpi-mgo-val').innerText = `$${mgo.toFixed(2)}`;
  }
  if (!isNaN(ifo)) {
    document.getElementById('kpi-ifo-val').innerText = `$${ifo.toFixed(2)}`;
  }
  if (!isNaN(hi5)) {
    document.getElementById('kpi-hi5-badge').innerText = `Hi5: $${Math.round(hi5)}/MT`;
  }
}

async function fetchForecastCones(marketSnapshot) {
  try {
    const payload = {
      BDI_Close: Number(marketSnapshot.BDI_Close),
      BDI_Open: Number(marketSnapshot.BDI_Open),
      BDI_High: Number(marketSnapshot.BDI_High),
      BDI_Low: Number(marketSnapshot.BDI_Low),
      Bunker_VLSFO: Number(marketSnapshot.Bunker_VLSFO),
      Bunker_MGO: Number(marketSnapshot.Bunker_MGO),
      Bunker_IFO380: Number(marketSnapshot.Bunker_IFO380),
      Hi5_Spread: Number(marketSnapshot.Hi5_Spread),
      BDI_7D_MA: Number(marketSnapshot.BDI_7D_MA),
      BDI_14D_MA: Number(marketSnapshot.BDI_14D_MA),
      BDI_30D_Vol: Number(marketSnapshot.BDI_30D_Vol),
    };

    const res = await apiFetch('/predict', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    });

    if (!res.ok) {
      const errDetail = await res.text();
      throw new Error(`Prediction API HTTP ${res.status}: ${errDetail}`);
    }

    const data = await res.json();
    console.log("API Forecast Payload:", data);

    state.forecasts = data.forecasts;

    // Update 30D Forecast KPI Card & Stats Strip with real ML forecast outputs
    const f30 = state.forecasts['30D'];
    if (f30) {
      document.getElementById('kpi-forecast-p50').innerText = Math.round(Number(f30.p50)).toLocaleString();
      document.getElementById('kpi-forecast-p10').innerText = Math.round(Number(f30.p10)).toLocaleString();
      document.getElementById('kpi-forecast-p90').innerText = Math.round(Number(f30.p90)).toLocaleString();

      const sentimentBadge = document.getElementById('kpi-sentiment-badge');
      if (sentimentBadge && data.trend_analysis) {
        const sentiment = data.trend_analysis.market_sentiment;
        sentimentBadge.innerText = sentiment;
        if (sentiment === 'Bullish') {
          sentimentBadge.className = 'kpi-badge up';
        } else if (sentiment === 'Bearish') {
          sentimentBadge.className = 'kpi-badge down';
        } else {
          sentimentBadge.className = 'kpi-badge neutral';
        }
      }

      // Update Stats Strip
      const stripP10 = document.getElementById('strip-p10');
      const stripP50 = document.getElementById('strip-p50');
      const stripP90 = document.getElementById('strip-p90');

      if (stripP10) stripP10.innerText = `${Math.round(Number(f30.p10)).toLocaleString()} pts`;
      if (stripP50) stripP50.innerText = `${Math.round(Number(f30.p50)).toLocaleString()} pts`;
      if (stripP90) stripP90.innerText = `${Math.round(Number(f30.p90)).toLocaleString()} pts`;
    }

    renderForecastChart();
  } catch (err) {
    console.error('CRITICAL: fetchForecastCones failed:', err);
  }
}

function renderForecastChart() {
  const canvas = document.getElementById('forecastChart');
  if (!canvas) {
    console.error("Canvas element 'forecastChart' not found in DOM.");
    return;
  }

  if (typeof Chart === 'undefined') {
    console.warn('Chart.js not yet loaded, retrying in 250ms...');
    setTimeout(renderForecastChart, 250);
    return;
  }

  if (!state.history || !state.history.length || !state.forecasts) {
    console.error('FATAL: Cannot render forecast chart - state data missing.', {
      historyLen: state.history?.length,
      hasForecasts: !!state.forecasts
    });
    return;
  }

  const ctx = canvas.getContext('2d');
  if (state.chartInstance) {
    state.chartInstance.destroy();
  }

  try {
    // 1. Map real historical dates & Close prices from feature store
    const histLabels = state.history.map(h => String(h.date).slice(5));
    const lastDateStr = state.history[state.history.length - 1].date;
    const lastDate = new Date(lastDateStr);
    const baseBdi = Number(state.history[state.history.length - 1].BDI_Close);

    // 2. Generate future dates for 7D, 14D, 30D
    const date7 = new Date(lastDate.getTime() + 7 * 86400000);
    const date14 = new Date(lastDate.getTime() + 14 * 86400000);
    const date30 = new Date(lastDate.getTime() + 30 * 86400000);

    const label7 = date7.toISOString().slice(5, 10) + ' (+7D)';
    const label14 = date14.toISOString().slice(5, 10) + ' (+14D)';
    const label30 = date30.toISOString().slice(5, 10) + ' (+30D)';

    let forecastLabels = [label7, label14, label30];
    if (state.selectedHorizon === '7') {
      forecastLabels = [label7];
    } else if (state.selectedHorizon === '14') {
      forecastLabels = [label7, label14];
    }

    const allLabels = [...histLabels, ...forecastLabels];
    const lastHistIndex = histLabels.length - 1;

    // 3. Historical series
    const histData = state.history.map(h => Number(h.BDI_Close));
    const paddedHist = new Array(allLabels.length).fill(null);
    for (let i = 0; i <= lastHistIndex; i++) {
      paddedHist[i] = histData[i];
    }

    // 4. Quantile projections starting exactly at last historical date
    const p10Data = new Array(allLabels.length).fill(null);
    const p50Data = new Array(allLabels.length).fill(null);
    const p90Data = new Array(allLabels.length).fill(null);

    p10Data[lastHistIndex] = baseBdi;
    p50Data[lastHistIndex] = baseBdi;
    p90Data[lastHistIndex] = baseBdi;

    // 7D Horizon
    if (state.forecasts['7D']) {
      p10Data[lastHistIndex + 1] = Number(state.forecasts['7D'].p10);
      p50Data[lastHistIndex + 1] = Number(state.forecasts['7D'].p50);
      p90Data[lastHistIndex + 1] = Number(state.forecasts['7D'].p90);
    }

    // 14D Horizon
    if (state.selectedHorizon !== '7' && state.forecasts['14D']) {
      p10Data[lastHistIndex + 2] = Number(state.forecasts['14D'].p10);
      p50Data[lastHistIndex + 2] = Number(state.forecasts['14D'].p50);
      p90Data[lastHistIndex + 2] = Number(state.forecasts['14D'].p90);
    }

    // 30D Horizon
    if (state.selectedHorizon === 'all' || state.selectedHorizon === '30') {
      const idx30 = state.selectedHorizon === '30' ? lastHistIndex + 1 : lastHistIndex + 3;
      if (state.forecasts['30D']) {
        p10Data[idx30] = Number(state.forecasts['30D'].p10);
        p50Data[idx30] = Number(state.forecasts['30D'].p50);
        p90Data[idx30] = Number(state.forecasts['30D'].p90);
      }
    }

    console.log("Chart Data:", allLabels, paddedHist, p50Data);

    // 5. Render Chart.js
    state.chartInstance = new Chart(ctx, {
      type: 'line',
      data: {
        labels: allLabels,
        datasets: [
          {
            label: 'Historical BDI',
            data: paddedHist,
            borderColor: '#00b4d8',
            backgroundColor: 'rgba(0, 180, 216, 0.08)',
            borderWidth: 2.2,
            pointRadius: 2.5,
            pointHoverRadius: 5,
            tension: 0.25,
            spanGaps: true,
          },
          {
            label: 'P90 (Bullish Boundary)',
            data: p90Data,
            borderColor: 'rgba(16, 185, 129, 0.9)',
            borderWidth: 2,
            borderDash: [5, 4],
            pointRadius: 4.5,
            pointBackgroundColor: '#10b981',
            fill: false,
            spanGaps: true,
            tension: 0.2,
          },
          {
            label: 'P10 (Downside Floor)',
            data: p10Data,
            borderColor: 'rgba(255, 77, 109, 0.9)',
            borderWidth: 2,
            borderDash: [5, 4],
            pointRadius: 4.5,
            pointBackgroundColor: '#ff4d6d',
            fill: false,
            spanGaps: true,
            tension: 0.2,
          },
          {
            label: 'P50 (Expected Forecast)',
            data: p50Data,
            borderColor: '#00f5d4',
            borderWidth: 3.2,
            pointRadius: 5.5,
            pointBackgroundColor: '#00f5d4',
            pointBorderColor: '#070d19',
            pointBorderWidth: 2,
            fill: false,
            spanGaps: true,
            tension: 0.2,
          }
        ]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        interaction: {
          mode: 'index',
          intersect: false,
        },
        plugins: {
          legend: {
            display: true,
            position: 'top',
            labels: {
              color: '#94a3b8',
              font: { family: 'Inter', size: 11 },
              usePointStyle: true,
              boxWidth: 8
            }
          },
          tooltip: {
            backgroundColor: 'rgba(7, 13, 25, 0.95)',
            titleColor: '#00f5d4',
            bodyColor: '#f0f6fc',
            borderColor: 'rgba(0, 245, 212, 0.4)',
            borderWidth: 1,
            padding: 10,
            callbacks: {
              label: function(context) {
                const val = context.parsed.y;
                if (val === null || val === undefined) return null;
                return `${context.dataset.label}: ${Math.round(val).toLocaleString()} pts`;
              }
            }
          }
        },
        scales: {
          x: {
            grid: { color: 'rgba(255, 255, 255, 0.04)' },
            ticks: { color: '#64748b', font: { family: 'JetBrains Mono', size: 10 }, maxTicksLimit: 14 }
          },
          y: {
            grid: { color: 'rgba(255, 255, 255, 0.06)' },
            ticks: { color: '#94a3b8', font: { family: 'JetBrains Mono', size: 10 } }
          }
        }
      }
    });
  } catch (chartErr) {
    console.error('CRITICAL: Error rendering Forecast Chart:', chartErr);
  }
}

async function triggerDefaultOptimization() {
  await runCharterOptimization();
}

async function runCharterOptimization() {
  const cargoVal = document.getElementById('input-cargo')?.value;
  const cargo = parseFloat(cargoVal) || 80000;
  const origin = document.getElementById('select-origin')?.value || 'Port Hedland';
  const destination = document.getElementById('select-destination')?.value || 'Dhamra';
  const mode = document.getElementById('select-mode')?.value || 'TimeCharter';
  const draftInputVal = document.getElementById('input-draft')?.value;
  const draftVal = parseFloat(draftInputVal) || null;

  const btn = document.getElementById('btn-optimize');
  if (btn) btn.innerHTML = '<i class="fa-solid fa-spinner fa-spin"></i> Solving Fleet Optimization...';

  try {
    const payload = {
      cargo_tonnes: cargo,
      origin_port: origin,
      destination_port: destination,
      custom_max_draft: draftVal,
      charter_mode: mode
    };

    const res = await apiFetch('/optimize', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    });

    if (!res.ok) {
      throw new Error(`Optimization API HTTP ${res.status}`);
    }

    const data = await res.json();
    state.currentOptimization = data;
    state.baselineOpt = data;

    renderOptimizationResults(data);
    updateStressTestResults();
  } catch (err) {
    console.error('Optimization request failed:', err);
  } finally {
    if (btn) btn.innerHTML = '<i class="fa-solid fa-wand-magic-sparkles"></i> Evaluate Optimal Fleet Allocation';
  }
}

function renderOptimizationResults(data) {
  if (!data) return;

  const recVessel = document.getElementById('rec-vessel-name');
  const recSummary = document.getElementById('rec-summary-text');
  const recCost = document.getElementById('rec-cost-val');

  if (recVessel) recVessel.innerText = `${data.recommended_vessel.toUpperCase()}`;
  if (recCost) recCost.innerText = `$${Number(data.optimal_landed_cost_pmt).toFixed(2)}`;
  if (recSummary) recSummary.innerText = data.recommendation_summary;

  // Update Benchmark KPI Card
  const kpiBenchmark = document.getElementById('kpi-benchmark-cost');
  const kpiRoute = document.getElementById('kpi-active-route');
  const vBadge = document.getElementById('kpi-vessel-badge');

  if (kpiBenchmark) kpiBenchmark.innerText = `$${Number(data.optimal_landed_cost_pmt).toFixed(2)}`;
  if (kpiRoute) kpiRoute.innerText = `${data.origin_port} → ${data.destination_port}`;
  if (vBadge) {
    vBadge.innerText = data.recommended_vessel;
    vBadge.className = data.optimization_status === 'OPTIMAL' ? 'kpi-badge up' : 'kpi-badge down';
  }

  // Populate comparison table
  const tbody = document.getElementById('vessel-matrix-body');
  if (!tbody) return;
  tbody.innerHTML = '';

  data.vessel_options.forEach(v => {
    const isWinner = v.vessel_class === data.recommended_vessel;
    const tr = document.createElement('tr');
    if (isWinner) tr.classList.add('winner-row');

    const draftBadge = v.is_feasible
      ? `<span class="badge-draft-ok"><i class="fa-solid fa-circle-check"></i> Clear (+${v.draft_clearance_m}m)</span>`
      : `<span class="badge-draft-warn"><i class="fa-solid fa-triangle-exclamation"></i> Exceeds (${Math.abs(v.draft_clearance_m)}m)</span>`;

    tr.innerHTML = `
      <td><strong>${v.vessel_class}</strong> ${isWinner ? '<i class="fa-solid fa-crown" style="color: var(--accent-cyan);"></i>' : ''}</td>
      <td>${Number(v.dwt).toLocaleString()} DWT</td>
      <td>${v.laden_draft_m} m</td>
      <td>${draftBadge}</td>
      <td>${v.num_voyages}</td>
      <td>${Number(v.duration_days).toFixed(1)} d</td>
      <td>${Number(v.vlsfo_consumption_mt).toFixed(1)} MT</td>
      <td>$${Number(v.fuel_cost_usd).toLocaleString()}</td>
      <td>$${Number(v.charter_hire_cost_usd).toLocaleString()}</td>
      <td>$${Number(v.total_landed_cost_usd).toLocaleString()}</td>
      <td><strong style="color: ${isWinner ? 'var(--accent-emerald)' : 'var(--text-primary)'}; font-family: var(--font-mono); font-size: 0.95rem;">$${Number(v.landed_cost_per_tonne).toFixed(2)}</strong></td>
      <td>${Number(v.co2_emissions_mt).toFixed(1)} MT</td>
    `;
    tbody.appendChild(tr);
  });
}

function updateStressTestResults() {
  const sliderBunker = document.getElementById('slider-bunker');
  const sliderBdi = document.getElementById('slider-bdi');
  if (!sliderBunker || !sliderBdi || !state.baselineOpt) return;

  const bunkerPct = parseFloat(sliderBunker.value);
  const bdiPct = parseFloat(sliderBdi.value);

  const baseVlsfo = state.baselineOpt.vlsfo_price_used;
  const baseBdi = state.baselineOpt.bdi_rate_used;

  const adjVlsfo = baseVlsfo * (1 + bunkerPct / 100);
  const adjBdi = baseBdi * (1 + bdiPct / 100);

  document.getElementById('slider-bunker-val').innerText = `${bunkerPct >= 0 ? '+' : ''}${bunkerPct}% ($${adjVlsfo.toFixed(0)}/MT)`;
  document.getElementById('slider-bdi-val').innerText = `${bdiPct >= 0 ? '+' : ''}${bdiPct}% (${Math.round(adjBdi)} pts)`;

  // Recalculate landed cost for Capesize, Panamax, Supramax
  state.baselineOpt.vessel_options.forEach(v => {
    const fuelCostPerTonne = (v.vlsfo_consumption_mt * adjVlsfo + v.mgo_consumption_mt * 760) / v.cargo_tonnes;
    const hireDaily = adjBdi * (v.vessel_class === 'Capesize' ? 13.8 : (v.vessel_class === 'Panamax' ? 8.5 : 7.2));
    const hireCostPerTonne = (hireDaily * v.duration_days) / v.cargo_tonnes;
    const portDuesPerTonne = v.port_dues_usd / v.cargo_tonnes;

    const totalPmt = fuelCostPerTonne + hireCostPerTonne + portDuesPerTonne;
    const basePmt = v.landed_cost_per_tonne;
    const deltaPct = ((totalPmt - basePmt) / basePmt) * 100;

    let targetValId, targetDeltaId;
    if (v.vessel_class === 'Capesize') {
      targetValId = 'stress-cape-val';
      targetDeltaId = 'stress-cape-delta';
    } else if (v.vessel_class === 'Panamax') {
      targetValId = 'stress-pana-val';
      targetDeltaId = 'stress-pana-delta';
    } else {
      targetValId = 'stress-supra-val';
      targetDeltaId = 'stress-supra-delta';
    }

    const valEl = document.getElementById(targetValId);
    const deltaEl = document.getElementById(targetDeltaId);
    if (valEl && deltaEl) {
      valEl.innerText = `$${totalPmt.toFixed(2)}`;
      deltaEl.innerText = `${deltaPct >= 0 ? '+' : ''}${deltaPct.toFixed(1)}% vs Base`;
      deltaEl.style.color = deltaPct > 0 ? 'var(--accent-coral)' : (deltaPct < 0 ? 'var(--accent-emerald)' : 'var(--text-muted)');
    }
  });
}

function renderSavedScenarios(scenarios) {
  const container = document.getElementById('scenario-grid-container');
  if (!container) return;

  container.innerHTML = '';
  if (!scenarios || !scenarios.length) {
    container.innerHTML = '<div style="color: var(--text-muted); font-size: 0.8rem;">No saved scenarios yet.</div>';
    return;
  }

  scenarios.forEach(sc => {
    const card = document.createElement('div');
    card.className = 'scenario-card';
    card.innerHTML = `
      <div>
        <div class="scenario-header">
          <span class="scenario-title">${sc.title || `${sc.origin} → ${sc.destination}`}</span>
          <span class="kpi-badge up">${sc.recommended_vessel}</span>
        </div>
        <div class="scenario-meta">
          <i class="fa-solid fa-route"></i> ${sc.origin} to ${sc.destination} &bull; ${Number(sc.cargo_tonnes).toLocaleString()} MT
        </div>
      </div>
      <div class="scenario-footer">
        <div>
          <span style="color: var(--text-muted); font-size: 0.7rem;">Landed Rate</span>
          <div style="font-family: var(--font-mono); font-weight: 700; color: var(--accent-emerald); font-size: 1.1rem;">
            $${Number(sc.optimal_landed_pmt).toFixed(2)}/MT
          </div>
        </div>
        <div style="text-align: right;">
          <span style="color: var(--text-muted); font-size: 0.7rem;">Total Voyage</span>
          <div style="font-family: var(--font-mono); font-size: 0.85rem;">
            $${(Number(sc.total_cost_usd) / 1e6).toFixed(2)}M
          </div>
        </div>
      </div>
    `;
    container.appendChild(card);
  });
}

async function saveCurrentScenario() {
  if (!state.currentOptimization) return;

  const opt = state.currentOptimization;
  const title = `${Number(opt.cargo_tonnes).toLocaleString()} MT ${opt.origin_port} to ${opt.destination_port}`;

  const payload = {
    title: title,
    cargo_tonnes: opt.cargo_tonnes,
    origin: opt.origin_port,
    destination: opt.destination_port,
    bdi_rate: opt.bdi_rate_used,
    vlsfo_price: opt.vlsfo_price_used,
    recommended_vessel: opt.recommended_vessel,
    optimal_landed_pmt: opt.optimal_landed_cost_pmt,
    total_cost_usd: opt.total_optimal_cost_usd,
    notes: opt.recommendation_summary
  };

  const btn = document.getElementById('btn-save-scenario');
  if (btn) btn.innerHTML = '<i class="fa-solid fa-spinner fa-spin"></i> Saving...';

  try {
    const res = await apiFetch('/history', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    });
    const saved = await res.json();
    if (saved && saved.scenario) {
      state.savedScenarios.unshift(saved.scenario);
      renderSavedScenarios(state.savedScenarios);
    }
  } catch (err) {
    console.error('Failed to save scenario:', err);
  } finally {
    if (btn) btn.innerHTML = '<i class="fa-solid fa-bookmark"></i> Save Current Scenario';
  }
}
