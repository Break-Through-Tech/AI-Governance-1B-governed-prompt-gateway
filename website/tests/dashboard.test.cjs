const assert = require('node:assert/strict');
const { readFileSync } = require('node:fs');
const { join } = require('node:path');
const { test } = require('node:test');
const { runInNewContext } = require('node:vm');

const app = readFileSync(join(__dirname, '../site/app.js'), 'utf8');

function dashboard(result) {
  const elements = Object.fromEntries(['view', 'timeline', 'stepbar'].map(id => [
    id, { innerHTML: '', addEventListener() {}, scrollIntoView() {} }
  ]));
  const handlers = {};
  const location = { hash: '#overview' };
  runInNewContext(app, {
    document: { getElementById: id => elements[id] },
    window: { addEventListener: (name, handler) => { handlers[name] = handler; } },
    navigator: {}, location
  });
  handlers['gateway-run']({ detail: { phase: 'complete', result } });
  return {
    stage(name) {
      location.hash = '#' + name;
      handlers.hashchange();
      return elements.view.innerHTML;
    },
    indicator(name, status, checked) {
      const item = elements.timeline.innerHTML.split('</li>').find(li => li.includes(`href="#${name}"`));
      assert.ok(item, 'Timeline link exists');
      assert.ok(item.includes(`class="tl-status">${status}<`), `Expected ${name} to show ${status}`);
      assert.equal(/class="[^"]*\bdone\b/.test(item), checked, `Checkmark state for ${name}`);
    }
  };
}

const entry = (node, decision, metrics = {}, reason = '') => ({ node, decision, metrics, reason });
const safe = entry('screen_risk', 'safe');
const prepared = [
  safe, entry('optimize_prompt', 'full mode'), entry('verify_fidelity', 'pass')
];
const routed = [...prepared, entry('cache_lookup', 'miss'), entry('route_model', 'gpt-oss-120b')];
const audit = status => entry('finalize_audit', status);

for (const simulated of [true, false]) {
  test(simulated ? 'simulated answer uses simulation status' : 'live answer uses provider status after fallback', () => {
    const response = simulated ? '[Simulated response from gpt-oss-120b]' : 'Provider answer <with text>';
    const attempts = simulated ? [] : [entry('generate_response', 'error', { simulated: true })];
    const ui = dashboard({
      final_status: 'answered', risk_label: 'safe', original_tokens: 14, optimized_tokens: 13, model_input_tokens: 13,
      model_id: 'gpt-oss-120b', response, fidelity_passed: true,
      decision_log: [...routed, ...attempts, entry('generate_response', 'ok', { simulated }), audit('answered')]
    });
    const status = simulated ? 'Simulated' : 'Live success';
    ui.indicator('overview', status, !simulated);
    ui.indicator('routing', status, !simulated);
    for (const stage of ['overview', 'routing']) {
      const html = ui.stage(stage);
      assert.ok(html.includes(simulated ? 'Simulated response' : 'Model response'));
      assert.ok(html.includes(response.replace('<', '&lt;').replace('>', '&gt;')));
      assert.equal(html.includes('No provider call was made'), simulated);
    }
    assert.match(ui.stage('overview'), new RegExp(`>Generation</span>\\s*</div>\\s*<span class="pill [^"]+">${status}<`));
  });
}

for (const [final_status, decision, status, response] of [
  ['blocked', 'unsafe', 'Blocked', 'Blocked by policy.'],
  ['redirected', 'redirect', 'Redirected', 'A licensed agent can help.']
]) {
  test(`${final_status} response shows safety outcome and skips downstream stages`, () => {
    const ui = dashboard({
      final_status, risk_label: decision, response,
      decision_log: [entry('screen_risk', decision), entry('handle_unsafe', final_status), audit(final_status)]
    });
    ui.indicator('overview', status, false);
    ui.indicator('risk', status, false);
    for (const stage of ['optimizer', 'cache', 'routing']) ui.indicator(stage, 'Skipped', false);
    ui.indicator('audit', 'Completed', true);
    for (const stage of ['overview', 'risk', 'routing']) assert.ok(ui.stage(stage).includes(response));
  });
}

test('a cache hit shows cached text while routing and generation are skipped', () => {
  const ui = dashboard({
    final_status: 'cached', risk_label: 'safe', cache_hit: true, response: 'Cached answer text',
    decision_log: [...prepared, entry('cache_lookup', 'hit'), audit('cached')]
  });
  ui.indicator('overview', 'Cache hit', true);
  ui.indicator('routing', 'Skipped', false);
  for (const stage of ['overview', 'routing']) {
    const html = ui.stage(stage);
    assert.ok(html.includes('Cached response') && html.includes('Cached answer text'));
    assert.ok(html.includes('Generation was skipped'));
  }
});

test('failed generation shows its audit error and no successful routing checkmark', () => {
  const ui = dashboard({
    final_status: 'failed', risk_label: 'safe', generation_error: 'generation failed',
    decision_log: [...routed, entry('generate_response', 'error_fallback', {}, 'Groq returned HTTP 403'), audit('failed')]
  });
  ui.indicator('overview', 'Failed', false);
  ui.indicator('routing', 'Failed', false);
  for (const stage of ['overview', 'routing']) {
    const html = ui.stage(stage);
    assert.ok(html.includes('Generation failed') && html.includes('Groq returned HTTP 403'));
  }
});

test('a result without audit evidence cannot mark stages complete', () => {
  const ui = dashboard({ final_status: 'answered', decision_log: [] });
  for (const stage of ['overview', 'risk', 'optimizer', 'cache', 'routing', 'audit']) {
    ui.indicator(stage, 'Skipped', false);
  }
  assert.ok(ui.stage('overview').includes('The gateway returned no response text.'));
});

test('failed fidelity and cache checks retain their own failure indicators', () => {
  const ui = dashboard({
    final_status: 'answered', fidelity_passed: false, response: 'Answer',
    decision_log: [
      ...prepared.slice(0, -1), entry('verify_fidelity', 'fail → using original prompt'),
      entry('cache_lookup', 'error_fallback'), entry('route_model', 'gpt-oss-120b'),
      entry('generate_response', 'ok', { simulated: false }), audit('answered')
    ]
  });
  ui.indicator('optimizer', 'Failed', false);
  ui.indicator('cache', 'Failed', false);
  ui.indicator('routing', 'Live success', true);
});


test('an answered audit without response text cannot display live success', () => {
  const ui = dashboard({
    final_status: 'answered', risk_label: 'safe',
    decision_log: [...routed, entry('generate_response', 'ok', { simulated: false }), audit('answered')]
  });
  ui.indicator('overview', 'Failed', false);
  ui.indicator('routing', 'Failed', false);
  assert.ok(ui.stage('routing').includes('The gateway returned no response text.'));
});


test('submissions replace the prior run and reject malformed results without rendering them', async () => {
  const integration = readFileSync(join(__dirname, '../site/live-integration.js'), 'utf8');
  const handlers = {};
  const events = [];
  const elements = new Map();
  function element(id) {
    if (!elements.has(id)) elements.set(id, {
      innerHTML: '', before() {}, focus() {}, addEventListener() {}, scrollIntoView() {},
      querySelector: element
    });
    return elements.get(id);
  }
  const calls = [];
  let reply;
  const context = {
    document: {
      getElementById: id => element('#' + id),
      querySelector: element,
      createElement: () => element('form-container')
    },
    window: {
      addEventListener: (name, handler) => { handlers[name] = handler; },
      dispatchEvent(event) {
        events.push(event.detail.phase);
        handlers[event.type](event);
      }
    },
    CustomEvent: class { constructor(type, { detail }) { this.type = type; this.detail = detail; } },
    navigator: {}, location: { hash: '#overview' },
    fetch: async (url, options) => {
      calls.push([url, JSON.parse(options.body).text]);
      return { ok: true, json: async () => reply };
    }
  };
  element('#live-form').addEventListener = (name, handler) => { handlers[name] = handler; };
  runInNewContext(app, context);
  runInNewContext(integration, context);

  async function submit(prompt, result) {
    element('#live-prompt').value = prompt;
    reply = result;
    const pending = handlers.submit({ preventDefault() {} });
    assert.match(element('#view').innerHTML, /Waiting for the gateway/);
    assert.doesNotMatch(element('#view').innerHTML, /prior answer/);
    await pending;
    assert.equal(element('#live-run').disabled, false);
  }

  const firstPrompt = '<img src=x onerror=alert(1)> & text';
  await submit(firstPrompt, {
    risk_label: 'safe', final_status: 'answered', original_prompt: firstPrompt, response: 'prior answer',
    decision_log: [entry('generate_response', 'ok', { simulated: true }), audit('answered')]
  });
  assert.match(element('#view').innerHTML, /&lt;img.*&gt; &amp; text/);
  assert.doesNotMatch(element('#view').innerHTML, /<img/);
  assert.match(element('#view').innerHTML, /prior answer/);
  assert.match(element('#view').innerHTML, /N\/A/);

  for (const malformed of [null, { risk_label: 'safe', final_status: 'answered', decision_log: [null] }]) {
    await submit('Malformed response test', malformed);
    assert.equal(element('#live-status').textContent, 'Request failed');
    assert.match(element('#live-error').textContent, /invalid gateway result/);
    assert.doesNotMatch(element('#view').innerHTML, /prior answer/);
  }

  await submit('Fresh submission', {
    risk_label: 'unsafe', final_status: 'blocked', response: 'Fresh refusal',
    decision_log: [entry('screen_risk', 'unsafe'), audit('blocked')]
  });
  assert.match(element('#view').innerHTML, /Fresh refusal/);
  assert.equal(element('#live-error').hidden, true);
  assert.deepEqual(events, ['running', 'complete', 'running', 'failed', 'running', 'failed', 'running', 'complete']);
  assert.deepEqual(calls[0], ['/gateway/run', firstPrompt]);
});
