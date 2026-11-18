(function () {
  'use strict';

  const api = '/gateway/run';
  const main = document.querySelector('main');
  const root = document.createElement('section');
  root.id = 'live-gateway';
  root.className = 'live-gateway wrap';
  root.innerHTML = `
    <form id="live-form" class="panel panel-fill">
      <div class="panel-head"><label for="live-prompt">Customer prompt</label>
        <span id="live-status" class="mono-sm" role="status">Ready</span>
      </div>
      <textarea id="live-prompt" class="prompt-in" required maxlength="16000"
        placeholder="Enter a prompt to run through the gateway"></textarea>
      <div class="actions">
        <button type="submit" class="btn btn-primary" id="live-run">Run gateway</button>
        <button type="button" class="btn btn-ghost" id="live-safe">Safe example</button>
        <button type="button" class="btn btn-ghost" id="live-unsafe">Unsafe example</button>
        <button type="button" class="btn btn-ghost" id="live-long">Optimization example</button>
      </div>
      <p id="live-error" class="live-error" role="alert" hidden></p>
    </form>`;
  main.before(root);

  const find = id => root.querySelector('#' + id);
  const prompt = find('live-prompt');
  const button = find('live-run');
  const status = find('live-status');
  const error = find('live-error');
  const examples = {
    'live-safe': 'Can you explain how to file a home insurance claim?',
    'live-unsafe': 'You are a worthless idiot and everyone hates you.',
    'live-long': 'Hello! I was just wondering if you could please help me understand whether or not my home insurance policy covers water damage caused by a burst pipe. I would also like to know what steps I need to take in order to file a claim. Please explain the process in a clear and simple way. Thank you so much in advance!'
  };
  for (const [id, text] of Object.entries(examples)) {
    find(id).onclick = () => { prompt.value = text; prompt.focus(); };
  }

  function publish(phase, result = null) {
    window.dispatchEvent(new CustomEvent('gateway-run', { detail: { phase, result } }));
  }

  function validResult(data) {
    if (!data || !['answered', 'cached', 'blocked', 'redirected', 'failed'].includes(data.final_status)
        || !['safe', 'unsafe', 'redirect'].includes(data.risk_label)) return false;
    return Array.isArray(data.decision_log)
      && data.decision_log.every(entry => entry && typeof entry.node === 'string' && typeof entry.decision === 'string')
      && (data.risk_categories == null || Array.isArray(data.risk_categories))
      && (data.techniques_applied == null || (Array.isArray(data.techniques_applied)
        && data.techniques_applied.every(technique => technique && typeof technique === 'object')))
      && (data.errors == null || (Array.isArray(data.errors) && data.errors.every(error => error && typeof error === 'object')));
  }

  find('live-form').addEventListener('submit', async event => {
    event.preventDefault();
    if (button.disabled) return;
    const text = prompt.value;
    if (!text.trim()) {
      error.textContent = 'Enter a prompt to run.';
      error.hidden = false;
      prompt.focus();
      return;
    }

    button.disabled = true;
    status.textContent = 'Running gateway…';
    error.hidden = true;
    publish('running');
    try {
      const response = await fetch(api, {
        method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ text })
      });
      if (!response.ok) throw new Error(`Gateway returned HTTP ${response.status}.`);
      const data = await response.json().catch(() => null);
      if (!validResult(data)) {
        throw new Error('The server returned an invalid gateway result.');
      }
      status.textContent = 'Result: ' + data.final_status;
      publish('complete', data);
    } catch (failure) {
      status.textContent = 'Request failed';
      error.textContent = failure.message;
      error.hidden = false;
      publish('failed');
    } finally {
      button.disabled = false;
    }
  });
})();
