(function () {
  'use strict';

  // ---------- sample prompt ----------
  var SAMPLE = [
    'You are a customer service assistant for a regulated insurance provider.    You are always helpful, polite, and professional. You must always be polite and professional with customers.',
    'Answer only using the policy context provided. Do not give legal advice. If the user requests account changes, escalate to a licensed agent.',
    'Remember: never provide legal advice, and only answer from the provided policy context.',
    '',
    '',
    '',
    '### Policy context',
    '- **Home §4.2:** Water damage from a sudden burst pipe is covered up to the dwelling limit, less your deductible.',
    '- **FAQ:** To update your mailing address, sign in and open Profile settings.',
    '- **Auto §9.4:** Roadside assistance is available on auto policies with the Plus rider.',
    '- **Claims guide:** File within 60 days of the loss and include photos and a repair estimate.',
    '',
    '### Question',
    'Um, so basically I was just wondering, if it is not too much trouble, is water damage from a burst pipe in my kitchen covered, and in order to file a claim, what do I need to do? Thanks so much in advance!!'
  ].join('\n');

  // ---------- techniques (run in this order) ----------
  var TECHS = [
    { id: 'whitespace', name: 'Whitespace normalization', desc: 'Collapses repeated spaces, blank lines, indentation and trailing spaces.' },
    { id: 'formatting', name: 'Markup stripping', desc: 'Removes markdown headings, bold markers, rules and emoji.' },
    { id: 'filler', name: 'Filler phrase removal', desc: 'Drops hedges, greetings, sign-offs and filler words.' },
    { id: 'verbose', name: 'Concise rewriting', desc: 'Replaces wordy phrases with shorter equivalents.' },
    { id: 'dedup', name: 'Deduplication', desc: 'Removes sentences that repeat an earlier instruction.' },
    { id: 'prune', name: 'Context pruning', desc: 'Drops list items that share no keywords with the question.', risk: true },
    { id: 'telegraphic', name: 'Article dropping', desc: 'Removes “a”, “an” and “the” for a terse, telegraphic style.', risk: true }
  ];
  var TECH_BY_ID = {};
  TECHS.forEach(function (t) { TECH_BY_ID[t.id] = t; });
  function preset(all) {
    var on = {};
    TECHS.forEach(function (t) { on[t.id] = all === true ? true : all === false ? false : !t.risk; });
    return on;
  }

  var SPAN_RULES = {
    whitespace: [
      [/(?<= ) +/g, ''],
      [/[ \t]+(?=\n)/g, ''],
      [/(?<=\n)[ \t]+/g, ''],
      [/(?<=\n)\n+/g, ''],
      [/\t+/g, ' ']
    ],
    formatting: [
      [/^#{1,6}[ \t]+/gm, ''],
      [/\*\*|__/g, ''],
      [/`+/g, ''],
      [/^[ \t]*([-=*_])\1{2,}[ \t]*$/gm, ''],
      [/\p{Extended_Pictographic}️?\s?/gu, ''],
      [/[¶·]{2,}|(?:¶\s*)+/g, '']
    ],
    filler: [
      [/^\s*(?:hi|hello|hey)(?: there)?[!,.]*\s*/gim, ''],
      [/\b(?:um+|uh+|erm|hmm+)\b[,.]?\s*/gi, ''],
      [/\bso basically,?\s*/gi, ''],
      [/\bbasically,?\s*/gi, ''],
      [/\bI was (?:just )?wondering,?\s*(?:if|whether)?\s*/gi, ''],
      [/,?\s*(?:if )?(?:it is|it's) not too much trouble,?\s*/gi, ''],
      [/\bif you don't mind,?\s*/gi, ''],
      [/\s*\b(?:thanks|thank you)(?: so much| very much| a lot)?(?: in advance)?[!.]*/gi, ''],
      [/\bI hope (?:this|that) makes sense[.!]*\s*/gi, ''],
      [/\b(?:actually|literally|honestly|really|just|kindly),?\s+/gi, '']
    ],
    verbose: [
      [/\bin order to\b/gi, 'to'],
      [/\bdue to the fact that\b/gi, 'because'],
      [/\bat this point in time\b/gi, 'now'],
      [/\bin the event that\b/gi, 'if'],
      [/\b(?:is|are) able to\b/gi, 'can'],
      [/\bhas the ability to\b/gi, 'can'],
      [/\ba large number of\b/gi, 'many'],
      [/\bthe majority of\b/gi, 'most'],
      [/\bwith (?:regard|respect) to\b/gi, 'about'],
      [/\bfor the purpose of\b/gi, 'for'],
      [/\bprior to\b/gi, 'before'],
      [/\bmake sure that\b/gi, 'ensure'],
      [/\bwhether or not\b/gi, 'whether'],
      [/\beach and every\b/gi, 'every'],
      [/\bfirst and foremost\b/gi, 'first'],
      [/\b(?:it is important to note that|please note that|I would like you to|could you please)\s*/gi, ''],
      [/\bwhat do I need to do\b/gi, 'what are the steps']
    ],
    telegraphic: [
      [/\b(?:the|a|an)\s+(?=[A-Za-z§])/gi, '']
    ]
  };

  var STOP = {};
  ('a an the and or but if then so of to in on at by for with from as is are was were be been being it its this that these those ' +
   'i me my you your we our they them their he she his her what which who how when where why do does did doing done ' +
   'can could will would should must may might shall have has had just also only always very all any some such into about up out ' +
   'please need want there here than too own same other more most each few both while over under again further once').split(' ')
    .forEach(function (w) { STOP[w] = true; });
  function stem(w) { return w.length > 4 ? w.replace(/(?:ing|ed|es|s|e|d)$/, '') : w; }
  function keywords(text) {
    var set = {};
    (text.toLowerCase().match(/[a-z]+/g) || []).forEach(function (w) {
      if (!STOP[w] && w.length > 1) set[stem(w)] = true;
    });
    return Object.keys(set);
  }

  var STAGES = [
    { id: 'intent', label: 'Intent classified', route: 'overview' },
    { id: 'risk', label: 'Risk screen passed', route: 'risk' },
    { id: 'optimize', label: 'Optimize tokens', route: 'optimizer' },
    { id: 'cache', label: 'Cache lookup', route: 'cache' },
    { id: 'route', label: 'Route model', route: 'routing' },
    { id: 'respond', label: 'Respond', route: 'overview' }
  ];

  // ---------- state ----------
  var KEY = 'prompt-gateway-state-v2';
  var state = { prompt: SAMPLE, on: preset(), lastTech: null, showDiff: true, price: 3, qpd: 50000, stage: 2 };
  try {
    var saved = JSON.parse(localStorage.getItem(KEY) || 'null');
    if (saved) state = Object.assign(state, saved, { on: Object.assign(preset(), saved.on) });
  } catch (e) { /* storage unavailable */ }
  function persist() { try { localStorage.setItem(KEY, JSON.stringify(state)); } catch (e) {} }

  // ---------- token estimate ----------
  // Heuristic stand-in for a BPE tokenizer: common words ≈ 1 token, long words split,
  // digit runs in groups of 3, each punctuation run ≈ 1, newline runs ≈ 1.
  function estTokens(s) {
    var n = 0, m, re = /(\n+)|( {2,}|\t+)|([A-Za-z]+)|(\d+)|([^\sA-Za-z\d])\5*/g;
    while ((m = re.exec(s))) {
      if (m[3]) n += Math.ceil(m[3].length / 8);
      else if (m[4]) n += Math.ceil(m[4].length / 3);
      else n += 1;
    }
    return n;
  }

  // ---------- optimizer engine ----------
  // The prompt is split into units (sentences, or whole lines for list items). Each unit holds
  // pieces tagged keep / cut / add, so every change stays traceable to the technique that made it.
  function splitUnits(text) {
    var units = [];
    text.split(/(?<=\n)/).forEach(function (line) {
      if (/^[ \t]*(?:[-*•]|\d+[.)])\s/.test(line)) { units.push({ text: line, bullet: true }); return; }
      var start = 0, i = 0;
      while (i < line.length) {
        if ('.!?'.indexOf(line[i]) >= 0) {
          var j = i;
          while (j + 1 < line.length && '.!?'.indexOf(line[j + 1]) >= 0) j++;
          if (j + 1 === line.length || /\s/.test(line[j + 1])) {
            var k = j + 1;
            while (k < line.length && /\s/.test(line[k])) k++;
            units.push({ text: line.slice(start, k), bullet: false });
            start = k; i = k; continue;
          }
          i = j + 1; continue;
        }
        i++;
      }
      if (start < line.length) units.push({ text: line.slice(start), bullet: false });
    });
    var merged = [];
    units.forEach(function (u) {
      if (!u.text.trim() && merged.length) merged[merged.length - 1].text += u.text;
      else merged.push(u);
    });
    return merged.map(function (u) { return { bullet: u.bullet, pieces: [{ text: u.text, kind: 'keep' }] }; });
  }
  function liveText(u) {
    return u.pieces.filter(function (p) { return p.kind === 'keep' || p.kind === 'add'; }).map(function (p) { return p.text; }).join('');
  }
  function matchCase(src, repl) {
    return repl && /^[A-Z]/.test(src) ? repl[0].toUpperCase() + repl.slice(1) : repl;
  }
  function applySpan(units, techId, rules, changes) {
    units.forEach(function (u) {
      rules.forEach(function (rule) {
        var re = rule[0], repl = rule[1], out = [];
        u.pieces.forEach(function (p) {
          if (p.kind !== 'keep') { out.push(p); return; }
          var s = p.text, last = 0, m, hit = false;
          re.lastIndex = 0;
          while ((m = re.exec(s))) {
            if (!m[0]) { re.lastIndex++; continue; }
            var r = matchCase(m[0], repl);
            if (r === m[0]) continue;
            hit = true;
            if (m.index > last) out.push({ text: s.slice(last, m.index), kind: 'keep' });
            out.push({ text: m[0], kind: 'cut', tech: techId });
            if (r) out.push({ text: r, kind: 'add', tech: techId });
            changes.push({ from: m[0], to: r });
            last = m.index + m[0].length;
          }
          if (!hit) { out.push(p); return; }
          if (last < s.length) out.push({ text: s.slice(last), kind: 'keep' });
        });
        u.pieces = out;
      });
    });
  }
  function cutUnit(u, techId, changes) {
    changes.push({ from: liveText(u).trim() });
    var out = [];
    u.pieces.forEach(function (p, idx) {
      if (p.kind === 'add') { out.push({ text: p.text, kind: 'gone', tech: p.tech }); return; }
      if (p.kind !== 'keep') { out.push(p); return; }
      var isLast = !u.pieces.slice(idx + 1).some(function (q) { return q.kind === 'keep' || q.kind === 'add'; });
      var nl = isLast ? (p.text.match(/\n+$/) || [''])[0] : '';
      if (p.text.length > nl.length) out.push({ text: p.text.slice(0, p.text.length - nl.length), kind: 'cut', tech: techId });
      if (nl) out.push({ text: nl, kind: 'keep' });
    });
    u.pieces = out;
  }
  function applyDedup(units, changes) {
    var seen = [];
    units.forEach(function (u) {
      var kw = keywords(liveText(u));
      if (kw.length < 2) return;
      var dup = seen.some(function (prev) {
        var overlap = kw.filter(function (w) { return prev.indexOf(w) >= 0; }).length;
        return overlap / Math.min(kw.length, prev.length) >= 0.6 && Math.min(kw.length, prev.length) >= 3;
      });
      if (dup) cutUnit(u, 'dedup', changes); else seen.push(kw);
    });
  }
  function applyPrune(units, changes) {
    var q = null;
    for (var i = units.length - 1; i >= 0; i--) { if (liveText(units[i]).indexOf('?') >= 0) { q = units[i]; break; } }
    if (!q) for (var j = units.length - 1; j >= 0; j--) { if (liveText(units[j]).trim()) { q = units[j]; break; } }
    if (!q) return;
    var qk = keywords(liveText(q));
    if (qk.length < 2) return;
    units.forEach(function (u) {
      if (u === q || !u.bullet || !liveText(u).trim()) return;
      var kw = keywords(liveText(u));
      if (!kw.some(function (w) { return qk.indexOf(w) >= 0; })) cutUnit(u, 'prune', changes);
    });
  }
  function runPipeline(text, on) {
    var units = splitUnits(text), steps = [], prev = estTokens(text);
    TECHS.forEach(function (t) {
      if (!on[t.id]) return;
      var changes = [];
      if (SPAN_RULES[t.id]) applySpan(units, t.id, SPAN_RULES[t.id], changes);
      else if (t.id === 'dedup') applyDedup(units, changes);
      else if (t.id === 'prune') applyPrune(units, changes);
      var tok = estTokens(units.map(liveText).join(''));
      steps.push({ tech: t, before: prev, after: tok, changes: changes });
      prev = tok;
    });
    return { units: units, steps: steps, after: prev, text: units.map(liveText).join('') };
  }

  var memo = { key: null, val: null };
  function compute() {
    var key = state.prompt + '\u0000' + JSON.stringify(state.on) + '\u0000' + state.price + '\u0000' + state.qpd;
    if (memo.key === key) return memo.val;
    var base = runPipeline(state.prompt, state.on);
    var orig = estTokens(state.prompt);
    var stepById = {};
    base.steps.forEach(function (s) { stepById[s.tech.id] = s; });
    var potential = {};
    TECHS.forEach(function (t) {
      if (state.on[t.id]) return;
      var on = Object.assign({}, state.on); on[t.id] = true;
      potential[t.id] = base.after - runPipeline(state.prompt, on).after;
    });
    var saved = orig - base.after;
    var pct = orig ? saved / orig * 100 : 0;
    var status = !orig ? { label: 'No prompt', cls: 'st-wait' }
      : pct < 30 ? { label: 'Below target', cls: 'st-low' }
      : pct <= 50 ? { label: 'Within target', cls: 'st-in' }
      : { label: 'Above target · review quality', cls: 'st-high' };
    var log = [{ t: '14:02:07.118', op: 'risk_screen.pass', v: 'stage 2', cls: 'd-off' }];
    TECHS.forEach(function (t, i) {
      var s = stepById[t.id];
      log.push({ t: '14:02:07.' + (131 + i * 9), op: 'optimize.' + t.id + (s ? '' : '  skipped'), v: s ? '−' + (s.before - s.after) + ' tok' : '0 tok', cls: s ? 'd-on' : 'd-off' });
    });
    log.push({ t: '14:02:07.211', op: 'optimize.complete  ' + orig + ' → ' + base.after, v: pct.toFixed(1) + '%', cls: 'seg-keep' });
    memo.key = key;
    memo.val = {
      orig: orig, after: base.after, saved: saved, pct: pct, status: status, units: base.units,
      steps: base.steps, stepById: stepById, potential: potential, optText: base.text.trim(), log: log,
      savedDay: saved * state.qpd / 1e6 * state.price,
      activeCount: TECHS.filter(function (t) { return state.on[t.id]; }).length
    };
    return memo.val;
  }

  // ---------- helpers ----------
  function esc(s) {
    return String(s).replace(/[&<>"']/g, function (c) {
      return { '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c];
    });
  }
  function money(v) { return '$' + v.toFixed(v < 1 ? 4 : 2); }
  function fmt(n) { return Number(n).toLocaleString('en-US'); }
  function showWs(s) { return s.replace(/ /g, '·').replace(/\t/g, '→').replace(/\n/g, '↵\n'); }
  function snippet(s) { s = s.replace(/\s+/g, ' ').trim(); return s.length > 60 ? s.slice(0, 57) + '…' : s; }
  var ICON = {
    chev: '<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M9 6l6 6-6 6"/></svg>',
    check: '<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="#9FD9B8" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M5 12l5 5 9-10"/></svg>',
    tick: '<svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="3" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M5 12l5 5 9-10"/></svg>',
    arrow: '<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M5 12h14"/><path d="M13 6l6 6-6 6"/></svg>',
    reset: '<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M3 12a9 9 0 1 0 3-6.7"/><path d="M3 4v5h5"/></svg>',
    target: '<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><circle cx="12" cy="12" r="9"/><circle cx="12" cy="12" r="5"/><circle cx="12" cy="12" r="1"/></svg>',
    doc: '<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M14 3H6a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V9z"/><path d="M14 3v6h6"/></svg>',
    pen: '<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M12 20h9"/><path d="M16.5 3.5a2.1 2.1 0 0 1 3 3L7 19l-4 1 1-4z"/></svg>',
    compress: '<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true" style="color:var(--accent)"><path d="M4 14h6v6"/><path d="M20 10h-6V4"/><path d="M14 10l7-7"/><path d="M3 21l7-7"/></svg>',
    copy: '<svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><rect x="9" y="9" width="11" height="11" rx="2"/><path d="M5 15V5a2 2 0 0 1 2-2h10"/></svg>',
    sliders: '<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M4 6h9"/><path d="M17 6h3"/><circle cx="15" cy="6" r="2"/><path d="M4 12h3"/><path d="M11 12h9"/><circle cx="9" cy="12" r="2"/><path d="M4 18h11"/><path d="M19 18h1"/><circle cx="17" cy="18" r="2"/></svg>',
    steps: '<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M4 20h4v-4h4v-4h4V8h4"/></svg>',
    term: '<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M4 17l6-6-6-6"/><path d="M12 19h8"/></svg>',
    warn: '<svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M12 3l10 18H2z"/><path d="M12 10v5"/><path d="M12 18h.01"/></svg>',
    shield: '<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M12 3l8 3v6c0 5-3.5 8-8 9-4.5-1-8-4-8-9V6l8-3z"/></svg>',
    db: '<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><ellipse cx="12" cy="5.5" rx="8" ry="2.5"/><path d="M4 5.5v6c0 1.4 3.6 2.5 8 2.5s8-1.1 8-2.5v-6"/><path d="M4 11.5v6c0 1.4 3.6 2.5 8 2.5s8-1.1 8-2.5v-6"/></svg>',
    route: '<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><circle cx="6" cy="19" r="2"/><circle cx="18" cy="5" r="2"/><path d="M8 19h8a3 3 0 0 0 0-6H8a3 3 0 0 1 0-6h8"/></svg>',
    chat: '<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M21 12a8 8 0 0 1-11.6 7.1L4 20l1-4.6A8 8 0 1 1 21 12z"/></svg>'
  };

  // ---------- shared fragments ----------
  function header(o) {
    return '<header class="head"><div class="head-text">' +
      '<div class="crumbs"><span>Pipeline</span>' + ICON.chev + '<span>Query #Q-0001</span><span class="tag">Estimated tokens</span></div>' +
      '<h1>' + esc(o.title) + '</h1><p class="lede">' + esc(o.lede) + '</p></div>' +
      (o.actions ? '<div class="actions">' + o.actions + '</div>' : '') + '</header>';
  }
  function pipeline() {
    return '<ol class="pipe" aria-label="Pipeline progress">' + STAGES.map(function (s, i) {
      var cls = i < state.stage ? 'done' : i === state.stage ? 'cur' : 'todo';
      var mark = cls === 'done' ? ICON.check : '<span class="dot ' + (cls === 'cur' ? 'dot-cur' : 'dot-todo') + '"></span>';
      return '<li class="' + cls + '"' + (cls === 'cur' ? ' aria-current="step"' : '') + '><a href="#' + s.route + '">' + mark + esc(s.label) + '</a></li>';
    }).join('') + '</ol>';
  }
  function stat(label, val, unit, sub, accent) {
    return '<div class="stat"><span class="stat-label">' + esc(label) + '</span>' +
      '<span class="stat-val' + (accent ? ' accent' : '') + '">' + val + (unit ? '<small>' + unit + '</small>' : '') + '</span>' +
      (sub ? '<span class="stat-sub">' + sub + '</span>' : '') + '</div>';
  }
  function logBlock(rows) {
    return '<div class="log">' + rows.map(function (r) {
      return '<div><span class="ts">' + esc(r.t) + '</span><span class="op">' + esc(r.op) + '</span><span class="' + r.cls + '">' + esc(r.v) + '</span></div>';
    }).join('') + '</div>';
  }
  function panelTitle(icon, text) { return '<div class="title">' + icon + '<h2>' + esc(text) + '</h2></div>'; }

  // ---------- views ----------
  var views = {};

  // The optimizer renders a static shell once; updateOptimizer() refills the live regions so
  // typing in the prompt box never loses focus or cursor position.
  views.optimizer = function () {
    return header({
      title: 'Token optimizer',
      lede: 'Enter a prompt, then click techniques to apply them. Each one runs in order and shows exactly what it removed or rewrote.',
      actions: '<button type="button" class="btn btn-ghost" data-act="reset">' + ICON.reset + 'Reset to recommended</button>' +
        '<button type="button" class="btn btn-primary" data-act="continue-cache">Continue to cache lookup' + ICON.arrow + '</button>'
    }) + pipeline() +
    '<section class="stats" aria-label="Token metrics" id="opt-stats"></section>' +
    '<section class="panel" aria-label="Reduction against target" style="padding:20px" id="opt-band"></section>' +
    '<section class="panel" aria-label="Optimization techniques">' +
      '<div class="panel-head">' + panelTitle(ICON.sliders, 'Techniques') +
        '<div class="row-actions"><span class="mono-sm" id="tech-count"></span>' +
        '<button type="button" class="btn btn-ghost btn-sm" data-act="all">Apply all</button>' +
        '<button type="button" class="btn btn-ghost btn-sm" data-act="none">Clear all</button></div></div>' +
      '<div class="tgrid" id="opt-techs"></div>' +
    '</section>' +
    '<section class="cols" aria-label="Prompt comparison">' +
      '<article class="prompt"><div class="prompt-head">' + panelTitle(ICON.pen, 'Your prompt') +
        '<div class="row-actions"><span class="mono-sm" id="in-tok"></span>' +
        '<button type="button" class="btn btn-ghost btn-sm" data-act="sample">Load sample</button>' +
        '<button type="button" class="btn btn-ghost btn-sm" data-act="clear">Clear</button></div></div>' +
        '<div class="prompt-body prompt-body-fill">' +
          '<label class="sr-only" for="prompt-in">Prompt to optimize</label>' +
          '<textarea id="prompt-in" class="prompt-in" spellcheck="false" placeholder="Paste a system prompt, context and question…">' + esc(state.prompt) + '</textarea>' +
          '<p class="hint">Token counts are estimated with a heuristic tokenizer; real counts vary by model. Nothing you type leaves your browser.</p>' +
        '</div></article>' +
      '<article class="prompt"><div class="prompt-head">' + panelTitle(ICON.compress, 'Optimized prompt') +
        '<div class="row-actions" style="gap:4px"><span class="mono-sm" id="opt-tok"></span>' +
        '<button type="button" class="btn btn-ghost btn-sm" data-act="diff" id="diff-btn">Show changes</button>' +
        '<button type="button" class="icon-btn" data-act="copy" aria-label="Copy optimized prompt">' + ICON.copy + '</button></div></div>' +
        '<div class="prompt-body"><div class="sec-text" id="opt-opt"></div></div></article>' +
    '</section>' +
    '<div class="legend-row" id="opt-legend">' +
      '<span><span class="seg-cut chip">removed</span>Cut by an applied technique</span>' +
      '<span><span class="seg-add chip">rewritten</span>Replaced with a shorter form</span>' +
      '<span><span class="seg-cut chip hl">outlined</span>Changed by the technique you clicked last</span>' +
    '</div>' +
    '<div class="cols">' +
      '<section class="panel" aria-label="Step by step"><div class="panel-head">' + panelTitle(ICON.steps, 'Step by step') + '</div><div class="steps" id="opt-steps"></div></section>' +
      '<section class="panel" aria-label="Decision log" style="gap:12px"><div class="panel-head">' + panelTitle(ICON.term, 'Decision log') +
        '<a href="#audit" style="font-size:12.5px;display:inline-flex;align-items:center;min-height:32px">Open audit log</a></div><div id="opt-log"></div></section>' +
    '</div>';
  };

  function updateOptimizer() {
    var r = compute(), on = state.on;
    var $ = function (id) { return document.getElementById(id); };
    if (!$('opt-stats')) return;

    $('in-tok').textContent = '≈' + r.orig + ' tok';
    $('tech-count').textContent = r.activeCount + ' of ' + TECHS.length + ' applied';

    $('opt-techs').innerHTML = TECHS.map(function (t, i) {
      var isOn = !!on[t.id];
      var s = r.stepById[t.id];
      var delta = isOn ? (s.before - s.after) : r.potential[t.id];
      var deltaHtml = isOn
        ? '<span class="tb-delta d-on">−' + delta + '</span>'
        : '<span class="tb-delta d-off" title="Estimated saving if applied">' + (delta > 0 ? '−' + delta : '0') + '</span>';
      return '<button type="button" class="tbtn" data-tech="' + t.id + '" aria-pressed="' + isOn + '">' +
        '<span class="tb-top"><span class="tb-box">' + (isOn ? ICON.tick : '') + '</span>' +
        '<span class="tb-name">' + (i + 1) + '. ' + esc(t.name) + '</span>' + deltaHtml + '</span>' +
        '<span class="tb-desc">' + esc(t.desc) + '</span>' +
        (t.risk ? '<span class="risk">' + ICON.warn + 'Fidelity risk</span>' : '') +
      '</button>';
    }).join('');

    $('opt-stats').innerHTML =
      stat('Original input', '≈' + r.orig, 'tok') +
      stat('Optimized input', '≈' + r.after, 'tok') +
      stat('Reduction', r.pct.toFixed(1) + '%', '−' + r.saved + ' tok', '', true) +
      stat('Est. savings / day', '$' + r.savedDay.toFixed(2), '', fmt(state.qpd) + ' queries at $' + Number(state.price).toFixed(2) + ' / 1M tok');

    $('opt-band').innerHTML =
      '<div class="panel-head">' + panelTitle(ICON.target, 'Target band · 30–50% input reduction') +
      '<span class="pill ' + r.status.cls + '">' + r.status.label + '</span></div>' +
      '<div class="band"><div class="band-target"></div><div class="band-fill" style="width:' + (Math.min(r.pct, 80) / 80 * 100) + '%"></div></div>' +
      '<div class="ticks"><span>0%</span><span>20%</span><span>40%</span><span>60%</span><span>80%</span></div>';

    // "Show changes" renders removals inline (struck through) and rewrites highlighted;
    // off, it shows the clean prompt exactly as it would be sent.
    var diff = state.showDiff, optHtml = '', hasText = false;
    r.units.forEach(function (u) {
      u.pieces.forEach(function (p) {
        var hl = p.tech && p.tech === state.lastTech ? ' hl' : '';
        var title = p.tech ? ' title="' + esc(TECH_BY_ID[p.tech].name) + '"' : '';
        if (p.kind === 'keep') { optHtml += esc(p.text); if (p.text.trim()) hasText = true; }
        else if (p.kind === 'add') { optHtml += diff ? '<span class="seg-add' + hl + '"' + title + '>' + esc(p.text) + '</span>' : esc(p.text); hasText = true; }
        else if (p.kind === 'cut' && diff) optHtml += '<span class="seg-cut' + hl + '"' + title + '>' + esc(p.tech === 'whitespace' ? showWs(p.text) : p.text) + '</span>';
      });
    });
    $('opt-opt').innerHTML = hasText ? '<span class="seg-keep">' + optHtml + '</span>' : '<span class="sec-empty">Enter a prompt on the left to see the optimized version here.</span>';
    $('opt-tok').textContent = '≈' + r.after + ' tok';
    $('diff-btn').setAttribute('aria-pressed', String(diff));
    $('opt-legend').hidden = !diff;

    $('opt-steps').innerHTML = TECHS.map(function (t, i) {
      var s = r.stepById[t.id];
      if (!s) {
        var pot = r.potential[t.id];
        return '<div class="step off"><div class="step-head"><span class="step-num">' + (i + 1) + '</span><span class="step-name">' + esc(t.name) + '</span>' +
          '<span class="mono-sm">skipped</span></div><div class="step-note">' + (pot > 0 ? 'Not applied · could save ≈' + pot + ' tok' : 'Not applied · nothing to change in this prompt') + '</div></div>';
      }
      var chips = s.changes.slice(0, 6).map(function (c) {
        if (t.id === 'whitespace') return '<span class="chip seg-cut">' + esc(showWs(c.from)) + '</span>';
        if (c.to) return '<span class="chip seg-cut">' + esc(snippet(c.from)) + '</span><span class="muted" aria-hidden="true">→</span><span class="chip seg-add">' + esc(c.to) + '</span>';
        return '<span class="chip seg-cut">' + esc(snippet(c.from)) + '</span>';
      }).join('');
      var more = s.changes.length > 6 ? '<span class="mono-sm" style="align-self:center">+' + (s.changes.length - 6) + ' more</span>' : '';
      return '<div class="step on"><div class="step-head"><span class="step-num">' + (i + 1) + '</span><span class="step-name">' + esc(t.name) + '</span>' +
        '<span class="mono-sm">' + s.before + ' → ' + s.after + '</span><span class="delta d-on">−' + (s.before - s.after) + '</span></div>' +
        (s.changes.length ? '<div class="step-chips">' + chips + more + '</div>' : '<div class="step-note">Applied · no matches in this prompt</div>') + '</div>';
    }).join('');

    $('opt-log').innerHTML = logBlock(r.log);
  }

  views.overview = function () {
    var r = compute();
    var routed = state.stage >= 6;
    return header({ title: 'Overview', lede: 'End-to-end view of query #Q-0001 as it moves through the governed gateway.' }) +
      pipeline() +
      '<section class="stats">' +
        stat('Stages complete', Math.min(state.stage, STAGES.length) + ' / ' + STAGES.length, '') +
        stat('Input tokens', '≈' + r.after, 'tok', 'from ≈' + r.orig + ' original') +
        stat('Reduction', r.pct.toFixed(1) + '%', '', r.status.label, true) +
        stat('Est. savings / day', '$' + r.savedDay.toFixed(2), '', fmt(state.qpd) + ' queries / day') +
      '</section>' +
      '<div class="cols">' +
        '<section class="panel panel-fill">' + panelTitle(ICON.chat, 'Incoming prompt') +
          '<p class="quote">' + (state.prompt.trim() ? esc(state.prompt.length > 700 ? state.prompt.slice(0, 700) + '…' : state.prompt) : '<span class="muted">No prompt entered yet.</span>') + '</p>' +
          '<div class="rows">' +
            '<div class="row"><div class="row-main"><span class="row-title">Intent</span><span class="row-sub">claims.coverage_question · confidence 0.96 (sample)</span></div><span class="pill st-in">Classified</span></div>' +
          '</div>' +
        '</section>' +
        '<section class="panel panel-fill">' + panelTitle(ICON.route, 'Stage status') +
          '<div class="rows">' + STAGES.map(function (s, i) {
            var st = i < state.stage ? ['Done', 'st-in'] : i === state.stage ? ['In progress', 'st-high'] : ['Pending', 'st-wait'];
            return '<div class="row"><div class="row-main"><a class="row-title" href="#' + s.route + '" style="text-decoration:none">' + esc(s.label) + '</a></div><span class="pill ' + st[1] + '">' + st[0] + '</span></div>';
          }).join('') + '</div>' +
        '</section>' +
      '</div>' +
      (routed ? '<section class="panel panel-fill">' + panelTitle(ICON.chat, 'Response (sample)') +
        '<p class="quote">Yes — water damage from a sudden burst pipe is covered under your home policy (§4.2) up to the dwelling limit, less your deductible. To file a claim, submit it within 60 days of the loss and include photos and a repair estimate.</p></section>'
        : '');
  };

  views.risk = function () {
    var checks = [
      ['PII detection', 'No unmasked account numbers, SSNs or card data in the prompt.', 'Pass'],
      ['Prompt injection', 'No instruction-override patterns found in user or retrieved text.', 'Pass'],
      ['Topic policy', 'Claims and coverage questions are in scope for this profile.', 'Pass'],
      ['Legal advice request', 'Query asks about coverage, not legal interpretation.', 'Pass'],
      ['Account change request', 'No account changes requested; no escalation needed.', 'Pass']
    ];
    return header({ title: 'Risk screening', lede: 'Every query is screened against the policy profile before any tokens are optimized or sent to a model.',
      actions: '<a class="btn btn-primary" href="#optimizer">Go to token optimizer' + ICON.arrow + '</a>' }) +
      pipeline() +
      '<section class="stats">' + stat('Checks run', checks.length, '') + stat('Passed', checks.length, '', '', true) + stat('Blocked', 0, '') + stat('Screen latency', '13', 'ms') + '</section>' +
      '<section class="panel">' + '<div class="panel-head">' + panelTitle(ICON.shield, 'Screening results (sample)') + '<span class="pill st-in">Passed</span></div>' +
        '<div class="rows">' + checks.map(function (c) {
          return '<div class="row"><div class="row-main"><span class="row-title">' + c[0] + '</span><span class="row-sub">' + c[1] + '</span></div><span class="pill st-in">' + c[2] + '</span></div>';
        }).join('') + '</div></section>';
  };

  views.cache = function () {
    var r = compute();
    if (state.stage < 3) {
      return header({ title: 'Semantic cache', lede: 'Looks up semantically similar answered queries so repeat questions skip the model entirely.' }) +
        pipeline() +
        '<div class="empty">Waiting on the token optimizer.<br>Finish optimizing the prompt to run a cache lookup.<br><br><a class="btn btn-primary" href="#optimizer">Back to token optimizer' + ICON.arrow + '</a></div>';
    }
    var hits = [
      ['Does my home insurance cover a burst pipe?', 0.89, 'Different policy section cited · stale (41 days)'],
      ['How do I file a claim for water damage?', 0.86, 'Answers half the question'],
      ['Is flood damage covered under my home policy?', 0.71, 'Flood ≠ sudden pipe burst'],
      ['How long do I have to file a home claim?', 0.64, 'Partial overlap']
    ];
    return header({ title: 'Semantic cache', lede: 'Looks up semantically similar answered queries so repeat questions skip the model entirely.',
      actions: '<button type="button" class="btn btn-primary" data-act="continue-route">Continue to model routing' + ICON.arrow + '</button>' }) +
      pipeline() +
      '<section class="stats">' + stat('Lookup key', '≈' + r.after, 'tok') + stat('Best similarity', '0.89', '') + stat('Hit threshold', '0.92', '') + stat('Result', 'Miss', '', 'Query forwarded to routing', true) + '</section>' +
      '<section class="panel"><div class="panel-head">' + panelTitle(ICON.db, 'Nearest cached queries (sample)') + '<span class="pill st-low">Below threshold</span></div>' +
        '<div class="rows">' + hits.map(function (h) {
          return '<div class="row"><div class="row-main"><span class="row-title">' + esc(h[0]) + '</span><span class="row-sub">' + esc(h[2]) + '</span></div>' +
            '<span class="meter" aria-hidden="true"><span style="width:' + h[1] * 100 + '%"></span></span><span class="mono-sm" style="min-width:36px;text-align:right">' + h[1].toFixed(2) + '</span></div>';
        }).join('') + '</div></section>';
  };

  views.routing = function () {
    var r = compute();
    if (state.stage < 4) {
      return header({ title: 'Model routing', lede: 'Chooses the cheapest model tier that meets the query’s quality and compliance requirements.' }) +
        pipeline() +
        '<div class="empty">Waiting on the cache lookup.<br>Routing runs after a cache miss.<br><br><a class="btn btn-primary" href="#cache">Go to semantic cache' + ICON.arrow + '</a></div>';
    }
    var tiers = [
      { name: 'Small · fast', mult: 0.1, note: 'Fails grounding check on multi-part claims questions', pick: false },
      { name: 'Medium · balanced', mult: 1, note: 'Meets grounding and citation requirements', pick: true },
      { name: 'Large · reasoning', mult: 5, note: 'Not needed: no multi-document reasoning', pick: false }
    ];
    return header({ title: 'Model routing', lede: 'Chooses the cheapest model tier that meets the query’s quality and compliance requirements.',
      actions: state.stage < 5 ? '<button type="button" class="btn btn-primary" data-act="respond">Send to model' + ICON.arrow + '</button>' : '<a class="btn btn-ghost" href="#overview">View response</a>' }) +
      pipeline() +
      '<section class="panel"><div class="panel-head">' + panelTitle(ICON.route, 'Candidate tiers') + '<span class="mono-sm">≈' + r.after + ' input tok</span></div>' +
        '<div class="rows">' + tiers.map(function (t) {
          var price = state.price * t.mult;
          var cost = r.after / 1e6 * price;
          return '<div class="row"><div class="row-main"><span class="row-title">' + esc(t.name) + '</span><span class="row-sub">' + esc(t.note) + ' · $' + price.toFixed(2) + ' / 1M tok</span></div>' +
            '<span class="mono-sm">' + money(cost) + ' / query</span>' +
            '<span class="pill ' + (t.pick ? 'st-in' : 'st-wait') + '">' + (t.pick ? 'Selected' : 'Skipped') + '</span></div>';
        }).join('') + '</div></section>' +
      '<section class="stats">' +
        stat('Cost / query (optimized)', money(r.after / 1e6 * state.price), '') +
        stat('Cost / query (original)', money(r.orig / 1e6 * state.price), '') +
        stat('Saved / day', '$' + r.savedDay.toFixed(2), '', '', true) +
      '</section>';
  };

  views.audit = function () {
    var r = compute();
    var rows = [
      { t: '14:02:07.096', op: 'request.received  web_chat', v: 'Q-0001', cls: 'd-off' },
      { t: '14:02:07.104', op: 'intent.classify  claims.coverage_question', v: '0.96', cls: 'd-off' }
    ].concat(r.log);
    if (state.stage >= 4) rows.push({ t: '14:02:07.239', op: 'cache.lookup  best=0.89 threshold=0.92', v: 'miss', cls: 'd-on' });
    if (state.stage >= 5) rows.push({ t: '14:02:07.244', op: 'route.select  tier=medium', v: r.after + ' tok', cls: 'seg-keep' });
    if (state.stage >= 6) rows.push({ t: '14:02:08.512', op: 'respond.complete', v: '200', cls: 'seg-keep' });
    return header({ title: 'Audit log', lede: 'Every gateway decision for query #Q-0001, in order. Entries update as you change optimizer techniques.',
      actions: '<button type="button" class="btn btn-ghost" data-act="copy-log">' + ICON.copy + 'Copy log</button>' }) +
      pipeline() +
      '<section class="panel">' + panelTitle(ICON.term, 'Events · ' + rows.length) + logBlock(rows) + '</section>';
  };

  // ---------- render & routing ----------
  var view = document.getElementById('view');
  function route() {
    var r = (location.hash || '').slice(1);
    return views[r] ? r : 'optimizer';
  }
  function render() {
    var r = route();
    view.innerHTML = views[r]();
    if (r === 'optimizer') updateOptimizer();
    document.querySelectorAll('.nav a').forEach(function (a) {
      var on = a.dataset.route === r;
      a.classList.toggle('on', on);
      if (on) a.setAttribute('aria-current', 'page'); else a.removeAttribute('aria-current');
    });
    document.title = (document.querySelector('h1') || {}).textContent + ' · Prompt Gateway';
  }
  function refresh() {
    persist();
    if (route() === 'optimizer') updateOptimizer(); else render();
  }
  var toastEl = document.getElementById('toast'), toastT;
  function toast(msg) {
    toastEl.textContent = msg; toastEl.classList.add('show');
    clearTimeout(toastT); toastT = setTimeout(function () { toastEl.classList.remove('show'); }, 1800);
  }
  function fallbackCopy(text) {
    var ta = document.createElement('textarea'); ta.value = text; ta.style.position = 'fixed'; ta.style.opacity = '0';
    document.body.appendChild(ta); ta.select();
    try { document.execCommand('copy'); } catch (e) {}
    ta.remove();
  }
  function copy(text, msg) {
    var done = function () { toast(msg); };
    if (navigator.clipboard && window.isSecureContext) {
      navigator.clipboard.writeText(text).then(done, function () { fallbackCopy(text); done(); });
    } else { fallbackCopy(text); done(); }
  }
  function setPrompt(text) {
    state.prompt = text;
    var ta = document.getElementById('prompt-in');
    if (ta) ta.value = text;
    refresh();
  }

  view.addEventListener('click', function (e) {
    var tb = e.target.closest('[data-tech]');
    if (tb) {
      var id = tb.dataset.tech;
      state.on[id] = !state.on[id];
      state.lastTech = state.on[id] ? id : null;
      refresh();
      var again = view.querySelector('[data-tech="' + id + '"]');
      if (again) again.focus();
      return;
    }
    var act = e.target.closest('[data-act]');
    if (!act) return;
    switch (act.dataset.act) {
      case 'reset': state.on = preset(); state.lastTech = null; refresh(); toast('Reset to recommended techniques'); break;
      case 'all': state.on = preset(true); state.lastTech = null; refresh(); break;
      case 'none': state.on = preset(false); state.lastTech = null; refresh(); break;
      case 'diff': state.showDiff = !state.showDiff; refresh(); break;
      case 'sample': setPrompt(SAMPLE); break;
      case 'clear': setPrompt(''); var ta = document.getElementById('prompt-in'); if (ta) ta.focus(); break;
      case 'continue-cache': state.stage = Math.max(state.stage, 3); persist(); location.hash = 'cache'; break;
      case 'continue-route': state.stage = Math.max(state.stage, 4); persist(); location.hash = 'routing'; break;
      case 'respond': state.stage = Math.max(state.stage, 6); persist(); location.hash = 'overview'; toast('Response delivered'); break;
      case 'copy': copy(compute().optText, 'Optimized prompt copied'); break;
      case 'copy-log': copy(Array.prototype.map.call(view.querySelectorAll('.log div'), function (d) { return d.innerText.replace(/\n/g, '  '); }).join('\n'), 'Audit log copied'); break;
    }
  });
  view.addEventListener('input', function (e) {
    if (e.target.id === 'prompt-in') { state.prompt = e.target.value; refresh(); }
  });

  var priceEl = document.getElementById('price'), qpdEl = document.getElementById('qpd');
  priceEl.value = state.price; qpdEl.value = state.qpd;
  function readNum(el, min, max, fallback) {
    var v = parseFloat(el.value);
    return isFinite(v) ? Math.min(max, Math.max(min, v)) : fallback;
  }
  priceEl.addEventListener('input', function () { state.price = readNum(priceEl, 0.1, 30, state.price); refresh(); });
  qpdEl.addEventListener('input', function () { state.qpd = Math.round(readNum(qpdEl, 100, 1000000, state.qpd)); refresh(); });
  [priceEl, qpdEl].forEach(function (el) { el.addEventListener('blur', function () { priceEl.value = state.price; qpdEl.value = state.qpd; }); });

  window.addEventListener('hashchange', function () { render(); window.scrollTo(0, 0); });
  render();
})();
