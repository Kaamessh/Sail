/**
 * Maritime Freight Intelligence & Prescriptive Chartering Decision Dashboard
 * Client Application Logic
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

// API Base URL (handles direct loading or serverless proxy)
const API_BASE = '/api';

// Initialize Dashboard
document.addEventListener('DOMContentLoaded', async () => {
  setupEventListeners();
  await checkSystemHealth();
  await loadMarketData();
  await triggerDefaultOptimization();
});

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

  // Charter Form Submit
  const charterForm = document.getElementById('charter-form');
  if (charterForm) {
    charterForm.addEventListener('submit', async (e) => {
      e.preventDefault();
      await runCharterOptimization();
    });
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

  // Origin Port Draft helper
  const originSelect = document.getElementById('select-origin');
  const destSelect = document.getElementById('select-destination');
  if (originSelect && destSelect) {
    originSelect.addEventListener('change', () => {
      checkDraftAlerts();
    });
    destSelect.addEventListener('change', () => {
      checkDraftAlerts();
    });
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
    const res = await fetch(`${API_BASE}/health`);
    const data = await res.json();
    const statusText = document.getElementById('system-status-text');
    const badge = document.getElementById('system-status-badge');

    if (data.status === 'healthy') {
      statusText.innerText = `ML Quantile Models Ready (${data.database.engine.includes('Supabase') ? 'Supabase' : 'Local DB'})`;
      badge.style.borderColor = 'rgba(16, 185, 129, 0.4)';
    } else {
      statusText.innerText = 'Models Degraded / Fallback Mode';
    }
  } catch (err) {
    console.warn('Health check error:', err);
  }
}

async function loadMarketData() {
  try {
    const res = await fetch(`${API_BASE}/history?limit=45`);
    const data = await res.json();
    state.history = data.history || [];
    state.currentMarket = data.latest_snapshot || {};
    state.savedScenarios = data.saved_scenarios || [];

    updateExecutiveKPIs(state.currentMarket);
    renderSavedScenarios(state.savedScenarios);

    // Call /api/predict for current snapshot
    await fetchForecastCones(state.currentMarket);
  } catch (err) {
    console.error('Failed to load market data:', err);
  }
}

function updateExecutiveKPIs(market) {
  if (!market) return;

  const bdiClose = Number(market.BDI_Close) || 1850;
  const bdi7dMa = Number(market.BDI_7D_MA) || Math.round(bdiClose * 0.995);
  const bdiVol = Number(market.BDI_30D_Vol) || 28.5;
  const vlsfo = Number(market.Bunker_VLSFO) || 585.0;
  const mgo = Number(market.Bunker_MGO) || 760.0;
  const ifo = Number(market.Bunker_IFO380) || 430.0;
  const hi5 = Number(market.Hi5_Spread) || (vlsfo - ifo);

  document.getElementById('kpi-bdi-val').innerText = Number(bdiClose).toLocaleString();
  document.getElementById('kpi-bdi-7dma').innerText = Number(bdi7dMa).toLocaleString();
  document.getElementById('kpi-bdi-vol').innerText = Number(bdiVol).toFixed(1);

  document.getElementById('kpi-vlsfo-val').innerText = `$${Number(vlsfo).toFixed(2)}`;
  document.getElementById('kpi-mgo-val').innerText = `$${Number(mgo).toFixed(2)}`;
  document.getElementById('kpi-ifo-val').innerText = `$${Number(ifo).toFixed(2)}`;
  document.getElementById('kpi-hi5-badge').innerText = `Hi5: $${Number(hi5).toFixed(0)}/MT`;
}

async function fetchForecastCones(marketSnapshot) {
  try {
    const payload = {
      BDI_Close: Number(marketSnapshot.BDI_Close) || 1850,
      BDI_Open: marketSnapshot.BDI_Open ? Number(marketSnapshot.BDI_Open) : undefined,
      BDI_High: marketSnapshot.BDI_High ? Number(marketSnapshot.BDI_High) : undefined,
      BDI_Low: marketSnapshot.BDI_Low ? Number(marketSnapshot.BDI_Low) : undefined,
      Bunker_VLSFO: Number(marketSnapshot.Bunker_VLSFO) || 585,
      Bunker_MGO: Number(marketSnapshot.Bunker_MGO) || 760,
      Bunker_IFO380: Number(marketSnapshot.Bunker_IFO380) || 430,
    };

    const res = await fetch(`${API_BASE}/predict`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    });

    const data = await res.json();
    state.forecasts = data.forecasts;

    // Update 30D Forecast KPI Card & Stats Strip
    const f30 = state.forecasts && state.forecasts['30D'];
    if (f30) {
      document.getElementById('kpi-forecast-p50').innerText = Math.round(f30.p50).toLocaleString();
      document.getElementById('kpi-forecast-p10').innerText = Math.round(f30.p10).toLocaleString();
      document.getElementById('kpi-forecast-p90').innerText = Math.round(f30.p90).toLocaleString();

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

      // Update Forecast Stats Strip below chart
      const stripP10 = document.getElementById('strip-p10');
      const stripP50 = document.getElementById('strip-p50');
      const stripP90 = document.getElementById('strip-p90');

      if (stripP10) stripP10.innerText = `${Math.round(f30.p10).toLocaleString()} pts`;
      if (stripP50) stripP50.innerText = `${Math.round(f30.p50).toLocaleString()} pts`;
      if (stripP90) stripP90.innerText = `${Math.round(f30.p90).toLocaleString()} pts`;
    }

    renderForecastChart();
  } catch (err) {
    console.error('Error fetching forecasts:', err);
  }
}

function renderForecastChart() {
  const canvas = document.getElementById('forecastChart');
  if (!canvas) return;

  if (typeof Chart === 'undefined') {
    console.warn('Chart.js not yet loaded, retrying in 250ms...');
    setTimeout(renderForecastChart, 250);
    return;
  }

  if (!state.history || !state.history.length || !state.forecasts) {
    console.warn('Chart waiting for data...', { history: state.history?.length, forecasts: state.forecasts });
    return;
  }

  const ctx = canvas.getContext('2d');
  if (state.chartInstance) {
    state.chartInstance.destroy();
  }

  try {
    // 1. Build historical labels & data points
    const histLabels = state.history.map(h => {
      if (!h.date) return '';
      return String(h.date).slice(5); // 'MM-DD'
    });

    const lastDateStr = state.history[state.history.length - 1]?.date || '2026-08-30';
    let lastDate = new Date(lastDateStr);
    if (isNaN(lastDate.getTime())) {
      lastDate = new Date();
    }

    const baseBdi = Number(state.currentMarket?.BDI_Close) || Number(state.history[state.history.length - 1]?.BDI_Close) || 1850;

    // 2. Build forecast projection date labels
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

    // 4. Quantile projections starting exactly from the last historical point
    const p10Data = new Array(allLabels.length).fill(null);
    const p50Data = new Array(allLabels.length).fill(null);
    const p90Data = new Array(allLabels.length).fill(null);

    // Anchor at last historical day
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

    // 5. Render Chart.js with safe filling & spanGaps enabled
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
    console.error('Error in renderForecastChart:', chartErr);
  }
}

async function triggerDefaultOptimization() {
  await runCharterOptimization();
}

async function runCharterOptimization() {
  const cargo = parseFloat(document.getElementById('input-cargo').value) || 80000;
  const origin = document.getElementById('select-origin').value;
  const destination = document.getElementById('select-destination').value;
  const mode = document.getElementById('select-mode').value;
  const draftVal = parseFloat(document.getElementById('input-draft').value) || null;

  const btn = document.getElementById('btn-optimize');
  if (btn) btn.innerHTML = '<i class="fa-solid fa-spinner fa-spin"></i> Solving MILP Optimization...';

  try {
    const payload = {
      cargo_tonnes: cargo,
      origin_port: origin,
      destination_port: destination,
      custom_max_draft: draftVal,
      charter_mode: mode
    };

    const res = await fetch(`${API_BASE}/optimize`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    });

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

  // Banner update
  const recBanner = document.getElementById('recommendation-banner');
  const recVessel = document.getElementById('rec-vessel-name');
  const recSummary = document.getElementById('rec-summary-text');
  const recCost = document.getElementById('rec-cost-val');

  recVessel.innerText = `${data.recommended_vessel.toUpperCase()}`;
  recCost.innerText = `$${Number(data.optimal_landed_cost_pmt).toFixed(2)}`;
  recSummary.innerText = data.recommendation_summary;

  // Update Benchmark KPI Card
  document.getElementById('kpi-benchmark-cost').innerText = `$${Number(data.optimal_landed_cost_pmt).toFixed(2)}`;
  document.getElementById('kpi-active-route').innerText = `${data.origin_port} → ${data.destination_port}`;
  const vBadge = document.getElementById('kpi-vessel-badge');
  vBadge.innerText = data.recommended_vessel;
  vBadge.className = data.optimization_status === 'OPTIMAL' ? 'kpi-badge up' : 'kpi-badge down';

  // Populate comparison table
  const tbody = document.getElementById('vessel-matrix-body');
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

  const baseVlsfo = state.baselineOpt.vlsfo_price_used || 585;
  const baseBdi = state.baselineOpt.bdi_rate_used || 1850;

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
    const res = await fetch(`${API_BASE}/history`, {
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
