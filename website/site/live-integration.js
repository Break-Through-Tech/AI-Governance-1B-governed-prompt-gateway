(function () {
'use strict';
const API = '/gateway/run';
const root = document.createElement('section');
root.id='live-gateway';
root.className='live-gateway';
const main=document.querySelector('main .wrap') || document.querySelector('main');
if(!main){console.error('Live gateway: dashboard container not found');return;}
main.parentNode.insertBefore(root,main);
const esc=s=>String(s??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const fmt=n=>typeof n==='number'?n.toLocaleString(undefined,{maximumFractionDigits:4}):'—';
let last=null;
function step(node){return (last?.decision_log||[]).find(x=>x.node===node)}
function pill(s){return '<span class="live-pill">'+esc(s)+'</span>'}
function render(){
root.innerHTML=`
<div class="live-head"><div><div class="live-kicker">CONNECTED BACKEND · LIVE RESULTS</div><h2>Governed Prompt Gateway</h2><p>Submit a prompt to run the real LangGraph pipeline. Results below come from the API, not the sample dashboard.</p></div><span id="live-status" class="live-status">${last?'Last run complete':'Ready to test'}</span></div>
<form id="live-form"><label for="live-prompt">Customer prompt</label><textarea id="live-prompt" required maxlength="12000" placeholder="Ask an insurance question or test safety screening…">${esc(window.__gatewayDraft||'Can you explain how to file a home insurance claim?')}</textarea><div class="live-actions"><button type="submit" id="live-run">Run complete gateway</button><button type="button" id="live-safe">Safe example</button><button type="button" id="live-unsafe">Unsafe example</button><button type="button" id="live-long">Optimization example</button></div></form>
<div id="live-error" role="alert"></div><div id="live-results">${last?results(last):'<p class="live-empty">Run a prompt to see real safety, optimization, caching, routing, and audit results.</p>'}</div>
<div class="live-note"><strong>About the dashboard below:</strong> The original walkthrough remains available for design preview. Its figures and sample decisions are not the live API results. Generation is currently simulated; intent, semantic caching, and dynamic routing require further implementation.</div>`;
root.querySelector('#live-form').addEventListener('submit',run);
root.querySelector('#live-safe').onclick=()=>setPrompt('Can you explain how to file a home insurance claim?');
root.querySelector('#live-unsafe').onclick=()=>setPrompt('You are a worthless idiot and everyone hates you.');
root.querySelector('#live-long').onclick=()=>setPrompt('Hello! I was just wondering if you could please help me understand whether or not my home insurance policy covers water damage caused by a burst pipe. I would also like to know what steps I need to take in order to file a claim. Please explain the process in a clear and simple way. Thank you so much in advance!');
root.querySelector('#live-prompt').addEventListener('input',e=>window.__gatewayDraft=e.target.value);
}
function setPrompt(s){window.__gatewayDraft=s;root.querySelector('#live-prompt').value=s;}
function results(d){
const blocked=d.final_status==='blocked'||d.final_status==='redirected';
const reduction=d.original_tokens>0?(100*(d.original_tokens-(d.model_input_tokens??d.original_tokens))/d.original_tokens).toFixed(1):'0.0';
const entries=(d.decision_log||[]).map(e=>`<tr><td><code>${esc(e.node)}</code></td><td>${esc(e.decision)}</td><td>${esc(e.reason)}</td><td>${fmt(e.duration_ms)} ms</td></tr>`).join('');
const techniques=(d.techniques_applied||[]).filter(t=>(t.changes||[]).length).map(t=>pill(t.name)).join(' ')||'No changes';
return `<div class="live-metrics"><div><small>Final status</small><strong>${esc(d.final_status)}</strong></div><div><small>ML safety</small><strong>${esc(d.risk_label)}</strong></div><div><small>Tokens saved</small><strong>${fmt(d.tokens_saved)} (${reduction}%)</strong></div><div><small>Latency</small><strong>${fmt(d.latency_ms)} ms</strong></div></div>
<div class="live-grid">
<article><h3>Safety screening</h3><p><b>Decision:</b> ${esc(d.risk_label)} ${blocked?'🛑':'✓'}</p><p><b>Categories:</b> ${esc((d.risk_categories||[]).join(', ')||'None')}</p><p><b>Detector:</b> ${esc(step('screen_risk')?.reason||'Not executed')}</p><p class="live-muted">Binary detector label is not a calibrated risk probability.</p></article>
<article><h3>Token optimization & fidelity</h3>${blocked?'<p>Skipped: request blocked before optimization.</p>':`<p><b>Original:</b> ${fmt(d.original_tokens)} tokens</p><p><b>Model input:</b> ${fmt(d.model_input_tokens)} tokens</p><p><b>Fidelity:</b> ${d.fidelity_passed?'Passed':'Not passed'} (${fmt(d.fidelity_score)})</p><p><b>Applied:</b> ${techniques}</p><details><summary>Compare prompts</summary><p><b>Original</b></p><pre>${esc(d.original_prompt)}</pre><p><b>Optimized / sent to model</b></p><pre>${esc(d.prompt_for_model)}</pre></details>`}</article>
<article><h3>Cache</h3><p><b>Status:</b> ${blocked?'Skipped':d.cache_hit?'Hit':'Miss'}</p><p><b>Similarity:</b> ${blocked?'—':fmt(d.cache_similarity)}</p><p class="live-muted">Current backend supports exact-match reuse; semantic cache integration is pending.</p></article>
<article><h3>Routing & cost</h3><p><b>Model:</b> ${esc(d.model_id||'Not selected')}</p><p><b>Tier:</b> ${esc(d.model_tier||'—')}</p><p><b>Estimated input cost:</b> $${fmt(d.cost_estimate)}</p><p><b>Estimated total:</b> $${fmt(d.total_cost)}</p><p class="live-muted">Routing currently uses a configured default model.</p></article>
</div>
<article class="live-response"><h3>Response</h3><pre>${esc(d.response||'No response')}</pre><p class="live-muted">${blocked?'Safety refusal: no LLM invoked.':'Generation is simulated, not a live provider call.'}</p></article>
<article class="live-audit"><h3>Actual LangGraph audit trail</h3><div class="live-table-wrap"><table><thead><tr><th>Node</th><th>Decision</th><th>Reason</th><th>Duration</th></tr></thead><tbody>${entries}</tbody></table></div><details><summary>Full JSON (for debugging)</summary><pre>${esc(JSON.stringify(d,null,2))}</pre></details></article>`;
}
async function run(ev){
ev.preventDefault();const btn=root.querySelector('#live-run');const err=root.querySelector('#live-error');
btn.disabled=true;btn.textContent='Running ML + LangGraph…';err.textContent='';
try{
const text=root.querySelector('#live-prompt').value.trim();if(!text)throw Error('Please enter a prompt.');
const resp=await fetch(API,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({text})});
if(!resp.ok)throw Error(`API returned ${resp.status}. Confirm Uvicorn serves POST ${API}.`);
const data=await resp.json();if(!data.decision_log||!data.final_status)throw Error('Unexpected API response; expected LangGraph decision_log and final_status.');
last=data;root.querySelector('#live-results').innerHTML=results(data);root.querySelector('#live-status').textContent='Last run: '+data.final_status;
}catch(e){err.textContent='Could not run gateway: '+e.message;}finally{btn.disabled=false;btn.textContent='Run complete gateway';}
}
render();
})();
