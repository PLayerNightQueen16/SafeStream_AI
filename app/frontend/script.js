let total = 0, allowedCount = 0, removedCount = 0, confSum = 0;

function now() {
  const d = new Date();
  return d.getHours().toString().padStart(2, '0') + ':' +
    d.getMinutes().toString().padStart(2, '0');
}

async function analyze() {
  const text = document.getElementById('inputText').value.trim();
  if (!text) return;

  const btn = document.getElementById('analyzeBtn');
  btn.disabled = true;
  btn.innerHTML = '<span class="loading-dots">Analyzing</span>';

  const resultsWrap = document.getElementById('resultsWrap');
  resultsWrap.style.display = 'block';

  document.getElementById('explText').innerHTML =
    '<span class="loading-dots">Analyzing content</span>';
  document.getElementById('decisionBadge').className = 'decision-badge';
  document.getElementById('decisionBadge').textContent = '';
  document.getElementById('scoresContainer').innerHTML = '';

  try {
    const res = await fetch("/moderate", {
      method: "POST",
      headers: {
        "Content-Type": "application/json"
      },
      body: JSON.stringify({ text })
    });

    if (!res.ok) {
      throw new Error("Server error");
    }

    const result = await res.json();

    if (!result || !result.ai_scores) {
      throw new Error("Invalid response format");
    }

    renderResult(result, text);

  } catch (e) {
    console.error(e);
    document.getElementById('explText').textContent =
      'Error analyzing content. Check backend.';
  } finally {
    btn.disabled = false;
    btn.innerHTML = `
      <svg class="shield-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
        <path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"/>
      </svg> Analyze Content
    `;
  }
}

function renderResult(r, text) {
  const conf = Math.round(r.confidence * 100);

  const badge = document.getElementById('decisionBadge');
  const icons = { allow: '✓', flag: '⚠', remove: '✕', review: 'ℹ' };

  badge.className = 'decision-badge badge-' + r.decision;
  badge.textContent =
    (icons[r.decision] || '') + ' ' + r.decision.toUpperCase();

  document.getElementById('explText').textContent = r.explanation;
  document.getElementById('metaDecision').textContent = r.decision;
  document.getElementById('metaConf').textContent = conf + '%';

  const sc = r.ai_scores;

  const labels = ['Toxicity', 'Insult', 'Threat', 'Obscene'];
  const keys = ['toxicity', 'insult', 'threat', 'obscene'];

  document.getElementById('scoresContainer').innerHTML = keys.map((k, i) => {
    const val = sc[k] || 0;
    const pct = Math.round(val * 100);

    const cls =
      pct >= 60 ? 'fill-high' :
        pct >= 30 ? 'fill-mid' :
          'fill-low';

    return `
      <div class="score-row">
        <div class="score-header">
          <span>${labels[i]}</span>
          <span>${pct}%</span>
        </div>
        <div class="score-bar">
          <div class="score-fill ${cls}" style="width:${pct}%"></div>
        </div>
      </div>
    `;
  }).join('');

  /* STATS */
  total++;
  if (r.decision === 'allow') allowedCount++;
  if (r.decision === 'remove') removedCount++;

  confSum += r.confidence;

  document.getElementById('stat-total').textContent = total;
  document.getElementById('stat-allowed').textContent = allowedCount;
  document.getElementById('stat-removed').textContent = removedCount;

  document.getElementById('stat-allowed-pct').textContent =
    Math.round((allowedCount / total) * 100) + '% of total';

  document.getElementById('stat-removed-pct').textContent =
    Math.round((removedCount / total) * 100) + '% of total';

  document.getElementById('stat-conf').textContent =
    Math.round((confSum / total) * 100) + '%';

  /* HISTORY */
  const list = document.getElementById('historyList');
  const empty = list.querySelector('.empty-history');
  if (empty) empty.remove();

  const item = document.createElement('div');
  item.className = 'history-item';

  item.innerHTML = `
    <div>
      <div class="h-badge h-${r.decision}">
        ${(r.decision === 'allow' ? '✓ ' :
          r.decision === 'remove' ? '✕ ' : '⚠ ')
        + r.decision.toUpperCase()}
      </div>
      <div class="history-text">
        ${text.length > 80 ? text.slice(0, 80) + '…' : text}
      </div>
    </div>
    <div class="history-time">${now()}</div>
  `;

  list.prepend(item);
}

document.getElementById('inputText').addEventListener('keydown', e => {
  if (e.key === 'Enter' && (e.ctrlKey || e.metaKey)) analyze();
});