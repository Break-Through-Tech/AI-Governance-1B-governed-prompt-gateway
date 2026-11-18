'use strict';
const $ = id => document.getElementById(id);
const set = (id, value) => { $(id).textContent = value == null ? '—' : String(value); };
$('run').addEventListener('click', async () => {
  const text = $('prompt').value.trim();
  if (!text) { set('error', 'Enter a query first.'); return; }
  $('run').disabled = true; $('results').hidden = true;
  set('error', ''); set('status', 'Running gateway… first model load can take a while.');
  try {
    const res = await fetch('/gateway/run', {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify({text})});
    if (!res.ok) throw new Error(`Gateway returned HTTP ${res.status}. Check your Uvicorn terminal.`);
    const d = await res.json();
    if (!['safe','unsafe','redirect'].includes(d.risk_label)) throw new Error('Missing valid risk decision');
    set('risk', d.risk_label.toUpperCase()); $('risk').className = 'value '+(d.risk_label==='safe'?'safe':'unsafe');
    set('categories', (d.risk_categories||[]).join(', ') || 'No category assigned. Binary detector does not provide a calibrated probability.');
    set('intent', (d.intent||'unknown') + (d.intent_confidence===0?' (classifier unavailable)':''));
    const original=d.original_tokens, used=d.model_input_tokens;
    const pct=typeof original==='number'&&original>0&&typeof used==='number'?100*(original-used)/original:null;
    set('saving', pct==null?'N/A':pct.toFixed(1)+'%');
    set('tokens', original==null?'Not computed':`${original} original → ${used==null?'—':used} prepared`);
    const cache = (d.decision_log||[]).find(e=>e.node==='cache_lookup');
    const score = typeof d.cache_similarity==='number' ? ` · similarity ${d.cache_similarity.toFixed(4)}` : '';
    set('cache', !cache?'Skipped':cache.decision==='error_fallback'?'Unavailable':(d.cache_hit?'HIT':'MISS')+score);
    set('model', d.model_tier ? `${d.model_tier} · ${d.model_id||''}` : 'Not routed');
    set('cost', typeof d.total_cost==='number'?'$'+d.total_cost.toFixed(6)+' (estimated)':'Not computed');
    set('final', d.final_status||'unknown'); set('response',d.response||'No response');
    set('optimized',d.prompt_for_model||'No prepared prompt');
    $('audit').replaceChildren();
    for(const entry of d.decision_log||[]){const block=document.createElement('div');block.className='panel';
      const title=document.createElement('strong');title.textContent=`${entry.node||'Step'}: ${entry.decision||''}`;
      const note=document.createElement('p');note.textContent=entry.reason||'';block.append(title,note);$('audit').append(block);}
    set('json',JSON.stringify(d,null,2));$('results').hidden=false;set('status','Complete');
  }catch(e){set('error',e.message);set('status','Failed safely');}
  finally{$('run').disabled=false;}
});
