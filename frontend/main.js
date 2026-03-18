/* ── state ───────────────────────────────────────────────────────────────────*/
let predictionHistory = [];
let priceChartInstance = null;

/* ── clock ───────────────────────────────────────────────────────────────────*/
function updateClock() {
  document.getElementById('clock').textContent =
    new Date().toLocaleTimeString('en-IN', {hour12:false, timeZone:'Asia/Kolkata'});
}
updateClock(); setInterval(updateClock, 1000);

/* ── tab management ──────────────────────────────────────────────────────────*/
function showTab(name) {
  document.querySelectorAll('.tab-content').forEach(el => el.style.display = 'none');
  document.querySelectorAll('.nav-item').forEach(el => el.classList.remove('active'));
  document.getElementById('tab-' + name).style.display = 'block';
  document.querySelectorAll('.nav-item').forEach(el => {
    if (el.getAttribute('onclick') && el.getAttribute('onclick').includes(name))
      el.classList.add('active');
  });
  if (name === 'about') buildArchDiagram();
}

/* ── chart defaults ──────────────────────────────────────────────────────────*/
Chart.defaults.color = '#4a5568';
Chart.defaults.font.family = "'IBM Plex Mono', monospace";
Chart.defaults.font.size = 10;

const CHART_OPTIONS = {
  responsive: true,
  maintainAspectRatio: false,
  interaction: { mode: 'index', intersect: false },
  plugins: {
    legend: { display: false },
    tooltip: {
      backgroundColor: '#0f1115', borderColor: '#1e2330', borderWidth: 1,
      titleColor: '#4a5568', bodyColor: '#dde3f0', padding: 10,
      callbacks: { label: ctx => `${ctx.dataset.label}: ₹${ctx.parsed.y.toFixed(2)}` }
    }
  },
  scales: {
    x: { grid: { color: '#151820' }, ticks: { maxTicksLimit: 8 } },
    y: { grid: { color: '#151820' }, ticks: { callback: v => '₹' + v.toLocaleString() } }
  }
};

/* ── build price chart ───────────────────────────────────────────────────────*/
function buildPriceChart(labels, actual, predicted) {
  const ctx = document.getElementById('priceChart').getContext('2d');
  if (priceChartInstance) priceChartInstance.destroy();
  priceChartInstance = new Chart(ctx, {
    type: 'line',
    data: {
      labels,
      datasets: [
        { label:'Actual',    data:actual,    borderColor:'#38bdf8', borderWidth:1.5, pointRadius:0, tension:0.3, fill:false },
        { label:'Predicted', data:predicted, borderColor:'#00d4aa', borderWidth:1.5, pointRadius:0, tension:0.3, fill:false, borderDash:[5,3] }
      ]
    },
    options: CHART_OPTIONS
  });
}

/* ── fetch real metrics + chart data from backend ────────────────────────────*/
function getBaseUrl() {
  const raw = (document.getElementById('apiUrl')?.value || 'http://localhost:5000/predict').trim();
  return raw.replace(/\/predict$/, '');
}

async function loadDashboardData() {
  const base = getBaseUrl();
  try {
    // fetch metrics
    const mRes = await fetch(base + '/metrics');
    if (mRes.ok) {
      const m = await mRes.json();
      document.getElementById('m-r2').textContent   = m.r2.toFixed(4);
      document.getElementById('m-rmse').textContent = m.rmse.toFixed(2) + ' INR';
      document.getElementById('m-mae').textContent  = m.mae.toFixed(2) + ' INR';
      document.getElementById('m-mape').textContent = m.mape.toFixed(2) + '%';
    }
    // fetch real chart data
    const cRes = await fetch(base + '/chart-data');
    if (cRes.ok) {
      const c = await cRes.json();
      buildPriceChart(c.labels, c.actual, c.predicted);
    }
  } catch {
    // backend not up yet — charts stay as placeholder
  }
}

/* ── health check ────────────────────────────────────────────────────────────*/
async function checkHealth() {
  const badge = document.getElementById('modelStatus');
  if (!badge) return;
  badge.className = 'conf-badge';
  badge.textContent = '● CONNECTING...';
  try {
    const res = await fetch(getBaseUrl() + '/health');
    if (!res.ok) throw new Error('HTTP ' + res.status);
    const d = await res.json();
    badge.className = 'conf-badge connected';
    badge.textContent = d.model_loaded
      ? `● ONLINE · ${d.data_points} rows`
      : '● BACKEND UP · model not loaded';
    await loadDashboardData();
  } catch (e) {
    badge.className = 'conf-badge error';
    badge.textContent = '● OFFLINE · ' + e.message;
  }
}

/* ── prediction ──────────────────────────────────────────────────────────────*/
function setStatus(msg, type) {
  const el = document.getElementById('statusMsg');
  el.textContent = msg;
  el.className = type === 'err' ? 'status-err' : 'status-ok';
}

function showResult(price, dateStr) {
  document.getElementById('resultSection').style.display = 'block';
  document.getElementById('resultPrice').textContent = '₹' + price.toFixed(2);
  const d = new Date(dateStr);
  const label = d > new Date() ? 'future forecast' : 'historical prediction';
  document.getElementById('resultMeta').textContent =
    `${label} · ${d.toLocaleDateString('en-IN', {day:'2-digit', month:'short', year:'numeric'})}`;
  const band = price * 0.022;
  document.getElementById('confHigh').textContent = '₹' + (price + band).toFixed(2);
  document.getElementById('confLow').textContent  = '₹' + (price - band).toFixed(2);
  addToHistory(price, dateStr);
}

function addToHistory(price, dateStr) {
  predictionHistory.unshift({ price, date: dateStr, time: new Date().toLocaleTimeString('en-IN', {hour12:false}) });
  if (predictionHistory.length > 10) predictionHistory.pop();
  renderHistory();
}

function renderHistory() {
  const c = document.getElementById('historyContainer');
  if (!predictionHistory.length) {
    c.innerHTML = '<div class="empty-state">No predictions yet · run a prediction above</div>';
    return;
  }
  c.innerHTML = `
    <table class="history-table">
      <thead><tr><th>#</th><th>DATE</th><th>PREDICTED PRICE</th><th>TYPE</th><th>TIME</th></tr></thead>
      <tbody>
        ${predictionHistory.map((h, i) => {
          const d = new Date(h.date);
          const isFuture = d > new Date();
          const typeColor = isFuture ? 'color:var(--accent)' : 'color:var(--blue)';
          return `<tr>
            <td style="color:var(--muted)">${i+1}</td>
            <td>${d.toLocaleDateString('en-IN', {day:'2-digit', month:'short', year:'numeric'})}</td>
            <td class="green">₹${h.price.toFixed(2)}</td>
            <td style="${typeColor};font-size:10px;letter-spacing:.5px">${isFuture ? 'FORECAST' : 'HISTORICAL'}</td>
            <td style="color:var(--muted)">${h.time}</td>
          </tr>`;
        }).join('')}
      </tbody>
    </table>`;
}

async function runPrediction() {
  const date = document.getElementById('targetDate').value;
  const url  = document.getElementById('apiUrl').value.trim();
  if (!date) { setStatus('error · select a date first', 'err'); return; }
  if (!url)  { setStatus('error · enter backend URL', 'err'); return; }
  const btn = document.getElementById('predictBtn');
  btn.disabled = true;
  document.getElementById('spinner').style.display = 'block';
  setStatus('connecting to ' + url + ' ...', 'ok');
  try {
    const res = await fetch(url, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ date })
    });
    if (!res.ok) {
      const err = await res.json().catch(() => ({}));
      throw new Error(err.error || 'HTTP ' + res.status);
    }
    const data = await res.json();
    const price = data.predicted_price ?? data.price ?? data.close;
    if (price == null) throw new Error('response missing predicted_price field');
    showResult(Number(price), date);
    setStatus('prediction complete · ' + new Date().toLocaleTimeString('en-IN', {hour12:false}), 'ok');
  } catch (e) {
    setStatus('error · ' + e.message, 'err');
  } finally {
    btn.disabled = false;
    document.getElementById('spinner').style.display = 'none';
  }
}

/* ── architecture diagram ────────────────────────────────────────────────────*/
function buildArchDiagram() {
  const container = document.getElementById('archDiagram');
  if (!container || container.children.length) return;
  const layers = [
    { icon:'IN',  iconBg:'#071a2a', iconColor:'#38bdf8', name:'Input layer',     desc:'Shape: (batch, 60, 5) — OHLCV sequence',         detail:'60 timesteps × 5 features' },
    { icon:'L1',  iconBg:'#071a12', iconColor:'#00d4aa', name:'LSTM layer 1',    desc:'64 units · return_sequences=True · Dropout 0.2', detail:'output: (batch, 60, 64)' },
    { icon:'L2',  iconBg:'#071a12', iconColor:'#00d4aa', name:'LSTM layer 2',    desc:'32 units · Dropout 0.2',                         detail:'output: (batch, 32)' },
    { icon:'FC',  iconBg:'#1a1207', iconColor:'#f59e0b', name:'Dense layer',     desc:'1 unit · linear activation',                     detail:'output: (batch, 1)' },
    { icon:'OUT', iconBg:'#1a0710', iconColor:'#f43f5e', name:'Predicted close', desc:'Inverse-transformed via close_scaler',           detail:'₹ price (INR)' },
  ];
  layers.forEach((l, i) => {
    if (i > 0) {
      const arrow = document.createElement('div');
      arrow.className = 'arch-arrow';
      arrow.textContent = '↓';
      container.appendChild(arrow);
    }
    const div = document.createElement('div');
    div.className = 'arch-layer';
    div.innerHTML = `
      <div class="arch-icon" style="background:${l.iconBg};color:${l.iconColor};border:1px solid ${l.iconColor}22">${l.icon}</div>
      <div>
        <div class="arch-info-name">${l.name}</div>
        <div class="arch-info-desc">${l.desc}</div>
      </div>
      <div class="arch-right">${l.detail}</div>`;
    container.appendChild(div);
  });
}

/* ── init ────────────────────────────────────────────────────────────────────*/
checkHealth();