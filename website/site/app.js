(function () {
  'use strict';

  const ICON = {
    chev: '<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M9 6l6 6-6 6"/></svg>',
    tick: '<svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="3" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M5 12l5 5 9-10"/></svg>',
    arrow: '<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M5 12h14"/><path d="M13 6l6 6-6 6"/></svg>',
    doc: '<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M14 3H6a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V9z"/><path d="M14 3v6h6"/></svg>',
    compress: '<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true" style="color:var(--accent)"><path d="M4 14h6v6"/><path d="M20 10h-6V4"/><path d="M14 10l7-7"/><path d="M3 21l7-7"/></svg>',
    copy: '<svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><rect x="9" y="9" width="11" height="11" rx="2"/><path d="M5 15V5a2 2 0 0 1 2-2h10"/></svg>',
    term: '<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M4 17l6-6-6-6"/><path d="M12 19h8"/></svg>',
    shield: '<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M12 3l8 3v6c0 5-3.5 8-8 9-4.5-1-8-4-8-9V6l8-3z"/></svg>',
    db: '<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><ellipse cx="12" cy="5.5" rx="8" ry="2.5"/><path d="M4 5.5v6c0 1.4 3.6 2.5 8 2.5s8-1.1 8-2.5v-6"/><path d="M4 11.5v6c0 1.4 3.6 2.5 8 2.5s8-1.1 8-2.5v-6"/></svg>',
    route: '<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><circle cx="6" cy="19" r="2"/><circle cx="18" cy="5" r="2"/><path d="M8 19h8a3 3 0 0 0 0-6H8a3 3 0 0 1 0-6h8"/></svg>',
    chat: '<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M21 12a8 8 0 0 1-11.6 7.1L4 20l1-4.6A8 8 0 1 1 21 12z"/></svg>'
  };

  const steps = [
    ['overview', 'Overview'], ['risk', 'Risk screening'], ['optimizer', 'Token optimizer'],
    ['cache', 'Semantic cache'], ['routing', 'Model routing'], ['audit', 'Audit log']
  ];
  const view = document.getElementById('view');
  const timeline = document.getElementById('timeline');
  const stepbar = document.getElementById('stepbar');
  let phase = 'idle';
  let result = null;

  function escape(value) {
    return String(value ?? '').replace(/[&<>"']/g, c => ({
      '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;'
    })[c]);
  }

  const number = value => Number.isFinite(value) ? value.toLocaleString(undefined, { maximumFractionDigits: 4 }) : 'N/A';
  const money = value => Number.isFinite(value) ? '$' + value.toFixed(6) : 'N/A';
  const reduction = (original, revised) => original > 0 && Number.isFinite(revised)
    ? (100 * (original - revised) / original).toFixed(1) + '%' : 'N/A';
  const entries = node => result.decision_log.filter(entry => entry.node === node);
  const latest = node => entries(node).at(-1);
  const failed = entry => ['error', 'error_fallback'].includes(entry?.decision);

  function stat(label, value, sub = '') {
    return `<div class="stat"><span class="stat-label">${escape(label)}</span>
      <span class="stat-val">${escape(value)}</span>
      ${sub ? `<span class="stat-sub">${escape(sub)}</span>` : ''}</div>`;
  }

  function panel(icon, title, content) {
    return `<section class="panel panel-fill"><div class="title">${ICON[icon]}<h2>${escape(title)}</h2></div>${content}</section>`;
  }

  function row(title, detail = '', status = '') {
    const tone = statusTone(status);
    return `<div class="row"><div class="row-main"><span class="row-title">${escape(title)}</span>
      ${detail ? `<span class="row-sub">${escape(detail)}</span>` : ''}</div>
      ${status ? `<span class="pill ${tone}">${escape(status)}</span>` : ''}</div>`;
  }

  const quote = value => `<p class="quote">${escape(value ?? 'N/A')}</p>`;
  const empty = message => `<p class="empty">${escape(message)}</p>`;
  const stats = (...cards) => `<section class="stats">${cards.join('')}</section>`;
  const rows = items => `<div class="rows">${items.join('')}</div>`;

  function statusTone(status) {
    if (['Completed', 'Live success', 'Cache hit'].includes(status)) return 'st-in';
    if (['Failed', 'Blocked'].includes(status)) return 'st-low';
    if (['Simulated', 'Redirected'].includes(status)) return 'st-high';
    return 'st-wait';
  }

  function nodeStatus(node) {
    const entry = latest(node);
    if (!entry || entry.decision === 'skipped') return 'Skipped';
    if (failed(entry) || (node === 'verify_fidelity' && result.fidelity_passed === false)) return 'Failed';
    if (node === 'screen_risk') {
      if (entry.decision === 'unsafe') return 'Blocked';
      if (entry.decision === 'redirect') return 'Redirected';
    }
    if (node === 'generate_response') {
      if (entry.decision !== 'ok') return 'Unknown';
      if (result.generation_error || !hasResponse()) return 'Failed';
      if (entry.metrics?.simulated === true) return 'Simulated';
      if (entry.metrics?.simulated === false) return 'Live success';
      return 'Unknown';
    }
    return 'Completed';
  }

  function hasResponse() {
    return typeof result.response === 'string' && Boolean(result.response.trim());
  }

  function outcomeStatus() {
    const audit = latest('finalize_audit');
    if (!audit || failed(audit)) return nodeStatus('finalize_audit');
    switch (audit.decision) {
      case 'answered': return nodeStatus('generate_response');
      case 'cached': return nodeStatus('cache_lookup') === 'Completed' && result.cache_hit && hasResponse()
        ? 'Cache hit' : 'Unknown';
      case 'blocked': return 'Blocked';
      case 'redirected': return 'Redirected';
      case 'failed': return 'Failed';
      default: return 'Unknown';
    }
  }

  function generationMode() {
    return outcomeStatus() === 'Cache hit' ? 'Skipped on cache hit' : nodeStatus('generate_response');
  }

  function responsePanel() {
    const status = {
      blocked: 'Blocked', redirected: 'Redirected', cached: 'Cache hit'
    }[result.final_status] ?? nodeStatus('generate_response');
    const labels = {
      'Live success': 'Model response',
      Simulated: 'Simulated response',
      'Cache hit': 'Cached response',
      Blocked: 'Blocked request',
      Redirected: 'Redirected request'
    };
    const title = labels[status] ?? 'Gateway response';
    if (result.final_status === 'failed') {
      const attempt = latest('generate_response');
      return panel('chat', 'Generation failed', quote(
        (failed(attempt) ? attempt.reason : null)
          || result.generation_error || 'The gateway returned no answer.'
      ));
    }
    const note = status === 'Simulated' ? '<p class="hint">Simulation placeholder. No provider call was made.</p>'
      : status === 'Cache hit' ? '<p class="hint">Returned from the cache. Generation was skipped.</p>' : '';
    return panel('chat', title, note + quote(hasResponse() ? result.response : 'The gateway returned no response text.'));
  }

  function overview() {
    const stages = [
      ['Intent classification', 'classify_intent'], ['Risk screening', 'screen_risk'],
      ['Token optimization', 'optimize_prompt'], ['Fidelity verification', 'verify_fidelity'],
      ['Cache lookup', 'cache_lookup'], ['Model routing', 'route_model'],
      ['Generation', 'generate_response'], ['Audit', 'finalize_audit']
    ];
    const d = result;
    const price = latest('route_model')?.metrics?.input_price_per_million;
    const savings = generationMode() === 'Live success' && Number.isFinite(price)
      && Number.isFinite(d.original_tokens) && Number.isFinite(d.model_input_tokens)
      ? (d.original_tokens - d.model_input_tokens) * price / 1_000_000 : null;
    return responsePanel() + stats(
      stat('Final status', d.final_status, generationMode()),
      stat('Original tokens', number(d.original_tokens)),
      stat('Optimized tokens', number(d.optimized_tokens)),
      stat('Prepared prompt tokens', number(d.model_input_tokens)),
      stat('Prepared prompt reduction', reduction(d.original_tokens, d.model_input_tokens)),
      stat('Estimated input savings', money(savings), 'Per request at the selected model’s input rate')
    ) + `<div class="cols">${panel('chat', 'Submitted prompt', quote(d.original_prompt)
      + rows([row('Intent', d.intent ? `${d.intent} · confidence ${number(d.intent_confidence)}` : 'N/A')]))}
      ${panel('route', 'Stage status', rows(stages.map(([label, node]) => row(label, '', nodeStatus(node)))))}</div>`;
  }

  function risk() {
    const d = result;
    const entry = latest('screen_risk');
    const decision = { safe: 'Safe', unsafe: 'Blocked', redirect: 'Redirected' }[d.risk_label] ?? 'N/A';
    const categories = d.risk_categories;
    return stats(stat('Screening decision', decision), stat('Risk score', number(d.risk_score)),
      stat('Screening latency', Number.isFinite(entry?.duration_ms) ? `${number(entry.duration_ms)} ms` : 'N/A'))
      + panel('shield', 'Screening results', rows([
        row('Risk classification', d.risk_label ?? 'N/A'),
        row('Detected categories', categories?.length ? categories.join(', ') : (failed(entry) ? 'N/A' : 'None reported')),
        row('Decision explanation', entry?.reason ?? 'N/A')
      ]))
      + (d.risk_label !== 'safe' ? responsePanel() : '');
  }

  function optimizer() {
    const d = result;
    if (!latest('optimize_prompt')) return empty('Optimization was skipped for this execution.');
    const fidelity = latest('verify_fidelity');
    const techniques = (d.techniques_applied ?? []).filter(t => t.changes?.length || t.tokens_before !== t.tokens_after);
    const techniqueRows = techniques.map(t => row(t.name ?? t.id, `${number(t.tokens_before)} → ${number(t.tokens_after)} tokens`));
    return stats(stat('Original tokens', number(d.original_tokens)), stat('Optimized tokens', number(d.optimized_tokens)),
      stat('Optimization reduction', reduction(d.original_tokens, d.optimized_tokens)),
      stat('Prepared prompt tokens', number(d.model_input_tokens)))
      + `<div class="cols">${panel('doc', 'Original prompt', quote(d.original_prompt))}
        ${panel('compress', 'Optimized prompt', quote(d.optimized_prompt))}</div>`
      + panel('shield', 'Fidelity verification', rows([
        row('Result', d.fidelity_passed === true ? 'Passed' : d.fidelity_passed === false ? 'Failed' : 'N/A'),
        row('Similarity', number(d.fidelity_score)),
        row('Decision explanation', fidelity?.reason ?? 'N/A'),
        row('Optimization attempts', number(d.optimization_attempts))
      ]))
      + (d.prompt_for_model && d.prompt_for_model !== d.optimized_prompt
        ? panel('doc', 'Prepared prompt after fidelity verification', quote(d.prompt_for_model)) : '')
      + panel('compress', 'Applied techniques', techniqueRows.length ? rows(techniqueRows) : quote('No changes reported.'));
  }

  function cache() {
    const lookup = latest('cache_lookup');
    if (!lookup) return empty('Cache lookup was skipped for this execution.');
    const d = result;
    const status = failed(lookup) ? 'Unavailable' : d.cache_hit === true ? 'Hit' : d.cache_hit === false ? 'Miss' : 'N/A';
    const cards = [stat('Cache status', status)];
    if (Number.isFinite(d.cache_similarity)) cards.push(stat('Similarity', number(d.cache_similarity)));
    if (Number.isFinite(lookup.metrics?.min_similarity)) {
      cards.push(stat('Configured threshold', number(lookup.metrics.min_similarity)));
    }
    return stats(...cards)
      + panel('db', 'Cache lookup', rows([
        row('Backend', lookup.metrics?.backend ?? 'N/A'),
        row('Lookup explanation', lookup.reason ?? 'N/A'),
        row('Generation', d.cache_hit ? 'Skipped. The gateway reused the cached response.' : generationMode())
      ]));
  }

  function auditRows(events) {
    return rows(events.map(e => `<div class="row"><div class="row-main">
      <span class="row-title">${escape(e.node)} · ${escape(e.decision)}</span>
      <span class="row-sub">${escape(e.reason)}</span>
      ${e.metrics && Object.keys(e.metrics).length ? `<details class="audit-metrics"><summary>Metrics</summary><pre>${escape(JSON.stringify(e.metrics, null, 2))}</pre></details>` : ''}
      </div><span class="mono-sm">${Number.isFinite(e.duration_ms) ? number(e.duration_ms) + ' ms' : 'N/A'}</span></div>`));
  }

  function routing() {
    const d = result;
    const route = latest('route_model');
    if (!route) return responsePanel()
      + empty(d.cache_hit ? 'Routing and generation were skipped on a cache hit.' : 'Routing was skipped for this execution.');
    const attempts = d.decision_log.filter(e => ['route_model', 'generate_response'].includes(e.node));
    return responsePanel() + stats(stat('Selected model', d.model_id || 'N/A'), stat('Model tier', d.model_tier ?? 'N/A'),
      stat('Estimated input cost', money(d.model_id ? d.cost_estimate : null)), stat('Generation', generationMode()))
      + panel('route', 'Routing rationale', quote(route.reason))
      + panel('term', 'Routing and generation attempts', auditRows(attempts));
  }

  function audit() {
    const d = result;
    return stats(stat('Final status', d.final_status), stat('Latency', Number.isFinite(d.latency_ms) ? number(d.latency_ms) + ' ms' : 'N/A'),
      stat('Audit events', number(d.decision_log.length)))
      + panel('term', 'Decision log', auditRows(d.decision_log)
        + `<details class="audit-metrics"><summary>Full execution JSON</summary><pre>${escape(JSON.stringify(d, null, 2))}</pre></details>`)
      + (d.errors?.length ? panel('term', 'Execution errors', rows(d.errors.map(e => row(e.node, e.error, 'Failed')))) : '')
      + responsePanel();
  }

  const views = { overview, risk, optimizer, cache, routing, audit };
  const pageNodes = { risk: 'screen_risk', optimizer: 'optimize_prompt', cache: 'cache_lookup', routing: 'route_model', audit: 'finalize_audit' };

  function pageStatus(route) {
    if (route === 'overview') return outcomeStatus();
    const status = nodeStatus(pageNodes[route]);
    if (route === 'optimizer' && status === 'Completed') return nodeStatus('verify_fidelity');
    if (route === 'routing' && status === 'Completed') return nodeStatus('generate_response');
    return status;
  }

  function currentStep() {
    return Math.max(0, steps.findIndex(([id]) => '#' + id === location.hash));
  }

  function render() {
    const current = currentStep();
    const [id, title] = steps[current];
    const heading = `<header class="head"><div class="head-text">
      <div class="crumbs"><span>Pipeline</span>${ICON.chev}<span>${escape(result?.query_id ?? 'Gateway')}</span></div>
      <h1>${title}</h1></div>
      ${id === 'audit' && result && navigator.clipboard ? `<button type="button" id="copy-log" class="btn btn-ghost">${ICON.copy}Copy log</button>` : ''}
      </header>`;
    const message = {
      idle: 'Submit a prompt to see this stage’s execution.',
      running: 'Waiting for the gateway to finish.',
      failed: 'The request failed before a gateway result could be displayed.'
    }[phase];
    view.innerHTML = heading + (result ? views[id]() : empty(message));
    document.title = title + ' · Prompt Gateway';

    timeline.innerHTML = steps.map(([route, label], i) => {
      const status = result ? pageStatus(route) : '';
      const tone = statusTone(status);
      const checked = tone === 'st-in';
      return `<li class="${i === current ? 'cur ' : ''}${checked ? 'done' : ''} ${tone}">
        <a class="tl-node" href="#${route}" ${i === current ? 'aria-current="step"' : ''}>
          <span class="tl-dot">${checked ? ICON.tick : i + 1}</span>
          <span class="tl-label">${label}</span>${status ? `<span class="tl-status">${status}</span>` : ''}
        </a></li>`;
    }).join('');
    const previous = steps[current - 1], next = steps[current + 1];
    stepbar.innerHTML = `<div class="stepbar-in"><span class="stepbar-pos">Step ${current + 1} of ${steps.length} · ${title}</span>
      <div class="stepbar-btns">${previous ? `<a class="btn btn-ghost" href="#${previous[0]}">Back</a>` : ''}
      ${next ? `<a class="btn btn-primary" href="#${next[0]}">Next: ${next[1]} ${ICON.arrow}</a>`
        : '<a class="btn btn-primary" href="#overview">Back to overview</a>'}</div></div>`;
  }

  view.addEventListener('click', async event => {
    if (!event.target.closest('#copy-log') || !result) return;
    const toast = document.getElementById('toast');
    try {
      await navigator.clipboard.writeText(JSON.stringify(result.decision_log, null, 2));
      toast.textContent = 'Audit log copied';
    } catch {
      toast.textContent = 'Could not copy the audit log';
    }
    toast.classList.add('show');
    setTimeout(() => toast.classList.remove('show'), 1800);
  });

  window.addEventListener('gateway-run', event => {
    ({ phase, result } = event.detail);
    render();
  });
  window.addEventListener('hashchange', () => {
    render();
    view.scrollIntoView({ block: 'start' });
  });
  render();
})();
