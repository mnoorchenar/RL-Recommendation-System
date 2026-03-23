// -----------------------------------------------------------------------
// RL Recommendation System — Frontend JavaScript
// -----------------------------------------------------------------------

let lossChart   = null;
let pollTimer   = null;
let lastMetrics = null;

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
      document.getElementById('tab-' + btn.dataset.tab).classList.add('active');
      if (btn.dataset.tab === 'training') updateLossChart();
      if (btn.dataset.tab === 'metrics')  loadMetrics();
    });
  });
}

// -----------------------------------------------------------------------
// Polling for status updates
// -----------------------------------------------------------------------
function startPolling() {
  pollStatus();
  pollTimer = setInterval(pollStatus, 2500);
}

async function pollStatus() {
  try {
    const res  = await fetch('/api/status');
    const data = await res.json();

    // Dataset stats
    if (data.dataset) {
      setText('stat-users', fmtNum(data.dataset.n_users));
      setText('stat-items', fmtNum(data.dataset.n_items));
      setText('stat-train', fmtNum(data.dataset.n_train));
      setText('stat-test',  fmtNum(data.dataset.n_test));
    }

    // Header badge + training panel
    const t = data.training;
    const headerBadge = document.getElementById('status-badge');
    const trainBadge  = document.getElementById('train-status-text');

    if (t.running) {
      setBadge(headerBadge, 'Training…', 'badge-running');
      setBadge(trainBadge,  `Epoch ${t.epoch} / ${t.total_epochs}`, 'badge-running');
    } else if (t.complete) {
      setBadge(headerBadge, 'Ready', 'badge-complete');
      setBadge(trainBadge,  'Complete ✓', 'badge-complete');
      clearInterval(pollTimer);   // stop polling once done
    } else if (t.error) {
      setBadge(headerBadge, 'Error', 'badge-error');
      setBadge(trainBadge,  'Error', 'badge-error');
    } else {
      setBadge(headerBadge, 'Initialising…', 'badge-pending');
      setBadge(trainBadge,  'Pending', 'badge-pending');
    }

    const epoch = t.total_epochs ? `${t.epoch} / ${t.total_epochs}` : '—';
    setText('train-epoch',   epoch);
    setText('train-loss',    t.loss    != null ? t.loss.toFixed(5)    : '—');
    setText('train-epsilon', t.epsilon != null ? t.epsilon.toFixed(4) : '1.0');

    const pct = Math.round((t.progress || 0) * 100);
    document.getElementById('progress-bar').style.width = pct + '%';
    setText('progress-label', pct + '%');

    updateLossChart();

    if (t.metrics && t.metrics !== lastMetrics) {
      lastMetrics = t.metrics;
      renderMetrics(t.metrics);
    }

  } catch (_) { /* network hiccup – ignore */ }
}

// -----------------------------------------------------------------------
// Loss chart
// -----------------------------------------------------------------------
function initLossChart() {
  const ctx = document.getElementById('loss-chart').getContext('2d');
  lossChart = new Chart(ctx, {
    type: 'line',
    data: {
      labels:   [],
      datasets: [{
        label:           'Avg Loss per Epoch',
        data:            [],
        borderColor:     '#6c63ff',
        backgroundColor: 'rgba(108,99,255,.12)',
        borderWidth:     2,
        fill:            true,
        tension:         0.4,
        pointRadius:     5,
        pointBackgroundColor: '#6c63ff',
      }],
    },
    options: {
      responsive: true,
      animation:  { duration: 300 },
      plugins: {
        legend: { labels: { color: '#8888aa' } },
      },
      scales: {
        x: { ticks: { color: '#8888aa' }, grid: { color: '#2a2a4a' } },
        y: { ticks: { color: '#8888aa' }, grid: { color: '#2a2a4a' }, beginAtZero: false },
      },
    },
  });
}

async function updateLossChart() {
  if (!lossChart) return;
  try {
    const res  = await fetch('/api/training_history');
    const data = await res.json();
    if (!data.losses || data.losses.length === 0) return;
    lossChart.data.labels                    = data.losses.map((_, i) => `Ep ${i + 1}`);
    lossChart.data.datasets[0].data         = data.losses;
    lossChart.update('none');
  } catch (_) {}
}

// -----------------------------------------------------------------------
// Demo — load user list
// -----------------------------------------------------------------------
async function loadUsers() {
  try {
    const res  = await fetch('/api/users');
    const data = await res.json();
    const sel  = document.getElementById('user-select');
    sel.innerHTML = '<option value="">— Select a user —</option>';
    (data.users || []).forEach(uid => {
      const opt = document.createElement('option');
      opt.value       = uid;
      opt.textContent = `User ${uid}`;
      sel.appendChild(opt);
    });
  } catch (_) {}
}

// -----------------------------------------------------------------------
// Demo — get recommendations
// -----------------------------------------------------------------------
async function getRecommendations() {
  const uid = document.getElementById('user-select').value;
  if (!uid) { alert('Please select a user first.'); return; }

  const k   = document.getElementById('k-select').value;
  const btn = document.getElementById('rec-btn');
  const err = document.getElementById('rec-error');

  btn.disabled    = true;
  btn.textContent = 'Loading…';
  err.style.display = 'none';

  try {
    const res  = await fetch(`/api/recommend/${uid}?k=${k}`);
    const data = await res.json();

    if (data.error) {
      err.textContent   = data.error;
      err.style.display = 'block';
      return;
    }
    renderHistory(data.history);
    renderRecommendations(data.recommendations, data.epsilon);

  } catch (e) {
    err.textContent   = 'Request failed: ' + e.message;
    err.style.display = 'block';
  } finally {
    btn.disabled    = false;
    btn.textContent = 'Get Recommendations';
  }
}

function renderHistory(history) {
  const el  = document.getElementById('history-list');
  const cnt = document.getElementById('history-count');

  if (!history || history.length === 0) {
    el.innerHTML  = '<div class="empty-state">No history available for this user.</div>';
    cnt.textContent = '';
    return;
  }

  cnt.textContent = `${history.length} interactions`;
  el.className    = 'item-list';
  el.innerHTML    = history.map(item => `
    <div class="item-row">
      <div>
        <div class="item-title">${escHtml(item.title)}</div>
        <div class="item-genres">${genreBadges(item.genres)}</div>
      </div>
      <span class="${item.liked ? 'liked-yes' : 'liked-no'}">${item.liked ? '👍' : '👎'}</span>
    </div>
  `).join('');
}

function renderRecommendations(recs, epsilon) {
  const el   = document.getElementById('rec-list');
  const cnt  = document.getElementById('rec-count');
  const info = document.getElementById('rec-info');

  if (!recs || recs.length === 0) {
    el.innerHTML    = '<div class="empty-state">No recommendations returned.</div>';
    cnt.textContent = '';
    info.style.display = 'none';
    return;
  }

  cnt.textContent    = `${recs.length} items`;
  info.style.display = 'block';
  info.textContent   = `ε = ${epsilon}  (exploration rate — lower means more confident policy)`;

  el.className = 'rec-grid';
  el.innerHTML = recs.map((item, i) => `
    <div class="rec-card">
      <div class="rec-rank">#${i + 1}</div>
      <div class="rec-title">${escHtml(item.title)}</div>
      <div>${genreBadges(item.genres)}</div>
      <div class="rec-score">Q-score: ${item.score}</div>
    </div>
  `).join('');
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
      // resume polling
      clearInterval(pollTimer);
      pollTimer = setInterval(pollStatus, 2500);
      alert(`Training started for ${epochs} epoch(s).`);
    } else {
      alert(data.message || 'Could not start training.');
    }
  } catch (e) {
    alert('Error: ' + e.message);
  }
}

// -----------------------------------------------------------------------
// Metrics
// -----------------------------------------------------------------------
async function loadMetrics() {
  try {
    const res  = await fetch('/api/metrics');
    const data = await res.json();
    if (!data.message) renderMetrics(data);
  } catch (_) {}
}

function renderMetrics(metrics) {
  const el      = document.getElementById('metrics-grid');
  const entries = Object.entries(metrics);
  if (!entries.length) return;

  el.className = 'metrics-grid';
  el.innerHTML = entries.map(([name, val]) => {
    const cls = val >= 0.3 ? 'good' : val >= 0.1 ? 'mid' : 'low';
    return `
      <div class="metric-card">
        <div class="metric-name">${name}</div>
        <div class="metric-val ${cls}">${(val * 100).toFixed(1)}%</div>
      </div>
    `;
  }).join('');
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
  el.textContent = text;
  el.className   = 'badge ' + cls;
}

function fmtNum(n) {
  return n != null ? Number(n).toLocaleString() : '—';
}

function escHtml(str) {
  return String(str)
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;');
}

function genreBadges(genres) {
  if (!genres) return '';
  return genres.split('|')
    .map(g => `<span class="genre-badge">${escHtml(g.trim())}</span>`)
    .join('');
}
