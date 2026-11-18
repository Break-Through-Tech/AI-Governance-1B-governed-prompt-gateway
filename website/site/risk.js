(function () {
  'use strict';
  const form = document.getElementById('risk-form');
  const prompt = document.getElementById('prompt');
  const submit = document.getElementById('submit');
  const result = document.getElementById('result');
  const label = document.getElementById('label');
  const explanation = document.getElementById('explanation');
  const details = document.getElementById('details');
  const empty = document.getElementById('empty');
  const count = document.getElementById('count');
  const example = document.getElementById('example');
  const refreshCount = () => { if (count) count.textContent = prompt.value.length.toLocaleString() + ' / 16,000'; };
  example?.addEventListener('click', () => {
    prompt.value = 'Where is my claim payment? You people are useless idiots.';
    prompt.dispatchEvent(new Event('input'));
    prompt.focus();
  });
  const next = document.getElementById('next-step');
  const status = document.getElementById('toxicity-status');
  function state(value, recommendation = '') {
    if (next) next.textContent = recommendation;
    if (status) status.textContent = value;
  }
  let controller;
  function field(name, value) {
    const term = document.createElement('dt'), definition = document.createElement('dd');
    term.textContent = name; definition.textContent = value;
    details.append(term, definition);
  }
  prompt.addEventListener('input', function () {
    refreshCount();
    state('Not run');
    if (!submit.disabled) { result.hidden = true; if (empty) empty.hidden = false; }
  });
  form.addEventListener('submit', async function (event) {
    event.preventDefault();
    if (submit.disabled || !prompt.value.trim()) return;
    controller = new AbortController();
    const timeout = setTimeout(() => controller.abort(), 360000);
    const text = prompt.value;
    prompt.disabled = submit.disabled = true;
    if (empty) empty.hidden = true;
    if (example) example.disabled = true;
    result.hidden = false; result.dataset.state = 'loading';
    state('Running');
    label.textContent = 'Screening…'; explanation.textContent = 'Waiting for the model. First-time model downloads can take several minutes; no decision has been issued.';
    details.replaceChildren();
    const started = performance.now();
    try {
      const response = await fetch('/check', {method:'POST',
        headers:{'Content-Type':'application/json'}, body:JSON.stringify({text}), signal:controller.signal});
      if (!response.ok) throw new Error('Screening unavailable');
      const data = await response.json();
      if (!['safe','unsafe'].includes(data.label)) throw new Error('Invalid response');
      if (data.label === 'unsafe' && (typeof data.category !== 'string' ||
          typeof data.confidence !== 'number' || !Number.isFinite(data.confidence) ||
          data.confidence < 0 || data.confidence > 1)) throw new Error('Invalid category response');
      result.dataset.state = data.label;
      state(data.label === 'unsafe' ? 'Checked · flagged' : 'Checked · not flagged',
        data.label === 'unsafe' ? 'Suggested next step: manual review' : 'Suggested next step: continue policy checks');
      label.textContent = data.label === 'unsafe' ? 'Flagged for toxicity' : 'Not flagged for toxicity';
      explanation.textContent = data.label === 'unsafe'
        ? 'The detector returned the unsafe class. This screen recommends holding the request for review; it does not enforce downstream model routing.'
        : 'The detector returned the safe class. Other policy checks have not been performed by this screen.';
      field('Detector', 'Team ToxicChat binary detector');
      field('Decision rule', 'Higher of the two class logits (argmax). No adjustable numeric threshold is exposed by this API.');

      if (data.label === 'unsafe') {
        field('Category', data.category);
        field('Category confidence', `${(data.confidence * 100).toFixed(1)}%`);
      }
      field('Request duration', `${Math.round(performance.now() - started)} ms including network and model loading`);
    } catch (_) {
      result.dataset.state = 'error'; state('Check failed', 'Suggested next step: retry screening'); label.textContent = 'No screening result';
      explanation.textContent = 'The request failed or timed out. This prompt has not passed screening. Check that the API and model files are available, then retry.';
      details.replaceChildren();
    } finally {
      clearTimeout(timeout); prompt.disabled = submit.disabled = false;
      if (example) example.disabled = false;
    }
  });
  window.addEventListener('pagehide', () => controller?.abort());
}());
