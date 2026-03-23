// -----------------------------------------------------------------------
// RL Ad Recommendation System — Frontend JavaScript
// -----------------------------------------------------------------------

let lossChart   = null;
let ctrChart    = null;
let roasChart   = null;
let pollTimer   = null;
let analyticsLoaded = false;

// -----------------------------------------------------------------------
// Boot
// -----------------------------------------------------------------------
document.addEventListener('DOMContentLoaded', () => {
  setupTabs();
  initLossChart();
  loadUsers();
  startPolling();
});

// -----------------------------------------------------------------------
// Tabs
// -----------------------------------------------------------------------
function setupTabs() {
  document.querySelectorAll('.tab').forEach(btn => {
    btn.addEventListener('click', () => {
      document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
      document.querySelectorAll('.tab-content').forEach(s => s.classList.remove('active'));
      btn.classList.add('active');
      const id = 'tab-' + btn.dataset.tab;
      document.getElementById(id).classList.add('active');

      if (btn.dataset.tab === 'training')    updateLossChart();
      if (btn.dataset.tab === 'analytics')   loadAnalytics();
      if (btn.dataset.tab === 'advertisers') loadAdvertisers();
    });
  });
}

// -----------------------------------------------------------------------
// Polling
// -----------------------------------------------------------------------
function startPolling() {
  pollStatus();
  pollTimer = setInterval(pollStatus, 2500);
}

async function pollStatus() {
  try {
    const res  = await fetch('/api/status');
    const data = await res.json();

    // KPI bar
    if (data.dataset) {
      const d = data.dataset;
      setText('stat-users',       fmtNum(d.n_users));
      setText('stat-ads',         fmtNum(d.n_ads));
      setText('stat-impressions', fmtNum(d.impressions));
      setText('stat-ctr',         pct(d.ctr));
      setText('stat-cvr',         pct(d.cvr));
      setText('stat-ecpm',        '$' + (d.ecpm || 0).toFixed(3));
    }

    const t     = data.training;
    const hBadge = document.getElementById('status-badge');
    const tBadge = document.getElementById('train-status-text');

    if (t.running) {
      setBadge(hBadge, 'Training…',                'badge-running');
      setBadge(tBadge, `Epoch ${t.epoch}/${t.total_epochs}`, 'badge-running');
    } else if (t.complete) {
      setBadge(hBadge, 'Ready',      'badge-complete');
      setBadge(tBadge, 'Complete ✓', 'badge-complete');
      clearInterval(pollTimer);
    } else if (t.error) {
      setBadge(hBadge, 'Error', 'badge-error');
      setBadge(tBadge, 'Error', 'badge-error');
    } else {
      setBadge(hBadge, 'Initialising…', 'badge-pending');
      setBadge(tBadge, 'Pending',       'badge-pending');
    }

    setText('train-epoch',   t.total_epochs ? `${t.epoch} / ${t.total_epochs}` : '—');
    setText('train-loss',    t.loss    != null ? t.loss.toFixed(5)    : '—');
    setText('train-epsilon', t.epsilon != null ? t.epsilon.toFixed(4) : '1.0');

    const p = Math.round((t.progress || 0) * 100);
    document.getElementById('progress-bar').style.width = p + '%';
    setText('progress-label', p + '%');

    updateLossChart();

    if (t.metrics) renderMetrics(t.metrics);

  } catch (_) {}
}

// -----------------------------------------------------------------------
// Loss chart
// -----------------------------------------------------------------------
function initLossChart() {
  const ctx = document.getElementById('loss-chart').getContext('2d');
  lossChart = new Chart(ctx, {
    type: 'line',
    data: {
      labels: [],
      datasets: [{
        label: 'Avg Huber Loss',
        data: [],
        borderColor: '#7c6dfa',
        backgroundColor: 'rgba(124,109,250,.1)',
        borderWidth: 2, fill: true, tension: 0.4,
        pointRadius: 4, pointBackgroundColor: '#7c6dfa',
      }],
    },
    options: {
      responsive: true,
      animation: { duration: 300 },
      plugins: { legend: { labels: { color: '#7878a0' } } },
      scales: {
        x: { ticks: { color: '#7878a0' }, grid: { color: '#2a2a4a' } },
        y: { ticks: { color: '#7878a0' }, grid: { color: '#2a2a4a' } },
      },
    },
  });
}

async function updateLossChart() {
  if (!lossChart) return;
  try {
    const res  = await fetch('/api/training_history');
    const data = await res.json();
    if (!data.losses || !data.losses.length) return;
    lossChart.data.labels                = data.losses.map((_, i) => `Ep ${i+1}`);
    lossChart.data.datasets[0].data     = data.losses;
    lossChart.update('none');
  } catch (_) {}
}

// -----------------------------------------------------------------------
// Demo — users
// -----------------------------------------------------------------------
async function loadUsers() {
  try {
    const res  = await fetch('/api/users');
    const data = await res.json();
    const sel  = document.getElementById('user-select');
    sel.innerHTML = '<option value="">— Select a user —</option>';
    (data.users || []).forEach(uid => {
      const o = document.createElement('option');
      o.value = uid; o.textContent = `User ${uid}`;
      sel.appendChild(o);
    });
  } catch (_) {}
}

async function getRecommendations() {
  const uid = document.getElementById('user-select').value;
  if (!uid) { alert('Please select a user first.'); return; }

  const k    = document.getElementById('k-select').value;
  const hour = document.getElementById('hour-select').value;
  const dow  = document.getElementById('dow-select').value;
  const btn  = document.getElementById('rec-btn');
  const err  = document.getElementById('rec-error');

  btn.disabled = true; btn.textContent = 'Loading…';
  err.style.display = 'none';

  let url = `/api/recommend/${uid}?k=${k}`;
  if (hour !== '') url += `&hour=${hour}`;
  if (dow  !== '') url += `&dow=${dow}`;

  try {
    const res  = await fetch(url);
    const data = await res.json();

    if (data.error) {
      err.textContent = data.error; err.style.display = 'block'; return;
    }

    renderUserProfile(data.profile);
    renderHistory(data.history);
    renderAdFeed(data.recommendations, data.epsilon);

  } catch (e) {
    err.textContent = 'Request failed: ' + e.message; err.style.display = 'block';
  } finally {
    btn.disabled = false; btn.textContent = '📡 Serve Ads';
  }
}

function renderUserProfile(profile) {
  if (!profile || !profile.user_id && profile.user_id !== 0) return;
  const box = document.getElementById('user-profile-box');
  box.style.display = 'block';
  setText('prof-age',    profile.age_group || '—');
  setText('prof-gender', profile.gender    || '—');
  setText('prof-device', profile.device    || '—');
  const ints = document.getElementById('prof-interests');
  ints.innerHTML = (profile.interests || [])
    .map(i => `<span class="badge badge-interest">${escHtml(i)}</span>`)
    .join(' ');
}

function renderHistory(history) {
  const el  = document.getElementById('history-list');
  const cnt = document.getElementById('history-count');
  if (!history || !history.length) {
    el.innerHTML = '<div class="empty-state">No impression history found.</div>';
    cnt.textContent = ''; return;
  }
  cnt.textContent = `${history.length} impressions`;
  el.className    = 'item-list';
  el.innerHTML    = history.map(item => {
    const outcome = item.converted ? '💰' : item.clicked ? '👆' : '👁';
    const cls     = item.converted ? 'outcome-convert' : item.clicked ? 'outcome-click' : 'outcome-skip';
    return `
      <div class="history-row">
        <div>
          <div style="font-weight:500">${escHtml(item.headline)}</div>
          <div class="history-meta">
            ${escHtml(item.advertiser)} &nbsp;·&nbsp;
            <span class="badge badge-cat">${escHtml(item.category)}</span>
            <span class="badge badge-fmt" style="margin-left:4px">${escHtml(item.format)}</span>
          </div>
        </div>
        <span class="${cls}" title="${item.converted ? 'Converted' : item.clicked ? 'Clicked' : 'Skipped'}">${outcome}</span>
      </div>`;
  }).join('');
}

function renderAdFeed(recs, epsilon) {
  const el   = document.getElementById('rec-list');
  const cnt  = document.getElementById('rec-count');
  const info = document.getElementById('rec-info');

  if (!recs || !recs.length) {
    el.innerHTML = '<div class="empty-state">No ads returned.</div>';
    cnt.textContent = ''; info.style.display = 'none'; return;
  }

  cnt.textContent    = `${recs.length} ads`;
  info.style.display = 'block';
  info.textContent   = `ε = ${epsilon} (exploration rate)  ·  Ranked by DQN Q-score  ·  CTR/CVR from neural predictor`;

  el.className = 'ad-grid';
  el.innerHTML = recs.map((ad, i) => `
    <div class="ad-card">
      <div class="ad-rank">#${i+1} &nbsp;<span class="badge badge-fmt">${escHtml(ad.format)}</span></div>
      <div class="ad-headline">${escHtml(ad.headline)}</div>
      <div class="ad-advertiser">${escHtml(ad.advertiser)}</div>
      <div>${catBadge(ad.category)}</div>
      <div class="ad-metrics">
        <div class="ad-metric">CTR <span class="ctr-val">${pct(ad.p_click)}</span></div>
        <div class="ad-metric">CVR <span class="cvr-val">${pct(ad.p_convert)}</span></div>
        <div class="ad-metric">Bid <span class="val">$${ad.bid_price.toFixed(2)}</span></div>
        <div class="ad-metric">Q <span class="val">${ad.q_score}</span></div>
      </div>
    </div>
  `).join('');
}

// -----------------------------------------------------------------------
// Analytics
// -----------------------------------------------------------------------
async function loadAnalytics() {
  if (analyticsLoaded) return;
  try {
    const res  = await fetch('/api/analytics');
    const data = await res.json();

    // KPI cards
    const kpiEl = document.getElementById('analytics-kpis');
    kpiEl.className = 'kpi-grid wide';
    kpiEl.innerHTML = [
      ['Impressions', fmtNum(data.total_impressions)],
      ['Clicks',      fmtNum(data.total_clicks)],
      ['Conversions', fmtNum(data.total_conversions)],
      ['CTR',         pct(data.ctr)],
      ['CVR',         pct(data.cvr)],
      ['eCPM',        '$' + (data.ecpm || 0).toFixed(3)],
      ['Revenue',     '$' + (data.total_revenue || 0).toFixed(2)],
    ].map(([lbl, val]) => `
      <div class="kpi-card">
        <div class="kpi-val">${val}</div>
        <div class="kpi-lbl">${lbl}</div>
      </div>
    `).join('');

    // Category table
    const catWrap = document.getElementById('category-table-wrap');
    if (data.by_category && data.by_category.length) {
      const maxImp = Math.max(...data.by_category.map(r => r.impressions));
      catWrap.innerHTML = `
        <table class="data-table">
          <thead><tr>
            <th>Category</th><th>Impressions</th><th>Clicks</th>
            <th>CTR</th><th>Conversions</th><th>Revenue</th>
          </tr></thead>
          <tbody>
            ${data.by_category.sort((a,b) => b.ctr - a.ctr).map(r => `
              <tr>
                <td>${catBadge(r.category)}</td>
                <td>
                  <div class="bar-cell">
                    ${fmtNum(r.impressions)}
                    <div class="bar-bg"><div class="bar-fill" style="width:${(r.impressions/maxImp*100).toFixed(1)}%"></div></div>
                  </div>
                </td>
                <td>${fmtNum(r.clicks)}</td>
                <td style="color:var(--green)">${pct(r.ctr)}</td>
                <td>${fmtNum(r.conversions)}</td>
                <td>$${r.revenue.toFixed(2)}</td>
              </tr>
            `).join('')}
          </tbody>
        </table>`;
    }

    // CTR bar chart
    buildCTRChart(data.by_category || []);

    // Metrics (if available)
    await loadMetrics();
    analyticsLoaded = true;

  } catch (_) {}
}

function buildCTRChart(cats) {
  const ctx = document.getElementById('ctr-chart').getContext('2d');
  if (ctrChart) ctrChart.destroy();
  ctrChart = new Chart(ctx, {
    type: 'bar',
    data: {
      labels: cats.map(r => r.category),
      datasets: [{
        label: 'CTR',
        data:  cats.map(r => +(r.ctr * 100).toFixed(2)),
        backgroundColor: 'rgba(124,109,250,.7)',
        borderColor: '#7c6dfa', borderWidth: 1, borderRadius: 4,
      }],
    },
    options: {
      responsive: true,
      plugins: {
        legend: { labels: { color: '#7878a0' } },
        tooltip: { callbacks: { label: ctx => ctx.parsed.y.toFixed(2) + '%' } },
      },
      scales: {
        x: { ticks: { color: '#7878a0' }, grid: { color: '#2a2a4a' } },
        y: {
          ticks: { color: '#7878a0', callback: v => v + '%' },
          grid: { color: '#2a2a4a' },
        },
      },
    },
  });
}

async function loadMetrics() {
  try {
    const res  = await fetch('/api/metrics');
    const data = await res.json();
    if (!data.message) renderMetrics(data);
  } catch (_) {}
}

function renderMetrics(metrics) {
  const el = document.getElementById('metrics-grid');
  if (!el) return;
  const entries = Object.entries(metrics);
  if (!entries.length) return;

  el.className = 'metrics-grid';

  // Format category labels
  const formatLabel = k => {
    const special = { CTR: 'CTR', CVR: 'CVR', eCPM: 'eCPM ($)', Coverage: 'Coverage' };
    return special[k] || k;
  };
  const formatVal = (k, v) => {
    if (k === 'eCPM') return '$' + v.toFixed(3);
    return (v * 100).toFixed(1) + '%';
  };

  el.innerHTML = entries.map(([k, v]) => {
    const cls = v >= 0.25 ? 'good' : v >= 0.08 ? 'mid' : 'low';
    return `
      <div class="metric-card">
        <div class="metric-name">${formatLabel(k)}</div>
        <div class="metric-val ${cls}">${formatVal(k, v)}</div>
      </div>`;
  }).join('');
}

// -----------------------------------------------------------------------
// Advertisers
// -----------------------------------------------------------------------
async function loadAdvertisers() {
  try {
    const res  = await fetch('/api/advertisers');
    const data = await res.json();
    const advs = data.advertisers || [];

    const wrap = document.getElementById('advertiser-table-wrap');
    if (!advs.length) { wrap.innerHTML = '<div class="empty-state">No data yet.</div>'; return; }

    const maxROAS = Math.max(...advs.map(a => a.roas));
    wrap.innerHTML = `
      <table class="data-table">
        <thead><tr>
          <th>Advertiser</th><th>Impressions</th><th>Clicks</th>
          <th>CTR</th><th>Conversions</th><th>Spend ($)</th><th>ROAS</th>
        </tr></thead>
        <tbody>
          ${advs.sort((a,b) => b.roas - a.roas).map(a => `
            <tr>
              <td style="font-weight:500;color:var(--text)">${escHtml(a.advertiser)}</td>
              <td>${fmtNum(a.impressions)}</td>
              <td>${fmtNum(a.clicks)}</td>
              <td style="color:var(--green)">${pct(a.ctr)}</td>
              <td>${fmtNum(a.conversions)}</td>
              <td>$${a.spend.toFixed(2)}</td>
              <td>
                <div class="bar-cell">
                  <span style="color:var(--orange);font-weight:600">${a.roas.toFixed(1)}x</span>
                  <div class="bar-bg"><div class="bar-fill" style="width:${(a.roas/maxROAS*100).toFixed(1)}%"></div></div>
                </div>
              </td>
            </tr>
          `).join('')}
        </tbody>
      </table>`;

    buildROASChart(advs);

  } catch (_) {}
}

function buildROASChart(advs) {
  const ctx = document.getElementById('roas-chart').getContext('2d');
  if (roasChart) roasChart.destroy();
  const sorted = [...advs].sort((a,b) => b.roas - a.roas);
  roasChart = new Chart(ctx, {
    type: 'bar',
    data: {
      labels: sorted.map(a => a.advertiser),
      datasets: [{
        label: 'ROAS (x)',
        data:  sorted.map(a => a.roas),
        backgroundColor: sorted.map((_, i) =>
          `hsla(${200 + i * 15}, 80%, 60%, 0.75)`
        ),
        borderRadius: 4,
      }],
    },
    options: {
      responsive: true,
      plugins: {
        legend: { labels: { color: '#7878a0' } },
        tooltip: { callbacks: { label: ctx => ctx.parsed.y.toFixed(2) + 'x ROAS' } },
      },
      scales: {
        x: { ticks: { color: '#7878a0' }, grid: { color: '#2a2a4a' } },
        y: {
          ticks: { color: '#7878a0', callback: v => v + 'x' },
          grid: { color: '#2a2a4a' },
        },
      },
    },
  });
}

// -----------------------------------------------------------------------
// Training — retrain
// -----------------------------------------------------------------------
async function retrain() {
  const epochs = parseInt(document.getElementById('epoch-input').value, 10) || 5;
  try {
    const res  = await fetch('/api/train', {
      method:  'POST',
      headers: { 'Content-Type': 'application/json' },
      body:    JSON.stringify({ epochs }),
    });
    const data = await res.json();
    if (res.ok) {
      clearInterval(pollTimer);
      pollTimer = setInterval(pollStatus, 2500);
      analyticsLoaded = false;
      alert(`Training started for ${epochs} epoch(s).`);
    } else {
      alert(data.message || 'Could not start training.');
    }
  } catch (e) { alert('Error: ' + e.message); }
}

// -----------------------------------------------------------------------
// Helpers
// -----------------------------------------------------------------------
function setText(id, text) {
  const el = document.getElementById(id);
  if (el) el.textContent = text;
}
function setBadge(el, text, cls) {
  if (!el) return;
  el.textContent = text; el.className = 'badge ' + cls;
}
function fmtNum(n) { return n != null ? Number(n).toLocaleString() : '—'; }
function pct(v)    { return v != null ? (v * 100).toFixed(2) + '%' : '—'; }
function escHtml(s) {
  return String(s)
    .replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
}
function catBadge(cat) {
  return `<span class="badge badge-cat">${escHtml(cat || '—')}</span>`;
}
