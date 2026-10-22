/*
 * FallacyLens visual story
 * ------------------------
 * This page is intentionally self-contained: no build step, framework, or
 * external model endpoint is needed for the presentation animation.
 * The interactive lab uses transparent illustrative cues so it never pretends
 * that a hand-authored browser heuristic is the trained model's final score.
 */

const $ = (selector, root = document) => root.querySelector(selector);
const $$ = (selector, root = document) => [...root.querySelectorAll(selector)];

const body = document.body;
const prefersReducedMotion = window.matchMedia?.('(prefers-reduced-motion: reduce)').matches;

const playMotion = (element, keyframes, options) => {
  if (body.classList.contains('reduced-motion') || !element?.animate) return;
  element.animate(keyframes, options);
};

const toast = (message) => {
  const node = $('#toast');
  node.textContent = message;
  node.classList.add('show');
  window.clearTimeout(toast.timer);
  toast.timer = window.setTimeout(() => node.classList.remove('show'), 3300);
};

const scrollToId = (id) => {
  const target = document.getElementById(id);
  if (target) target.scrollIntoView({ behavior: body.classList.contains('reduced-motion') ? 'auto' : 'smooth' });
};

// Reveal sections as the presenter moves through the story.
const revealObserver = 'IntersectionObserver' in window
  ? new IntersectionObserver((entries) => {
      entries.forEach((entry) => {
        if (entry.isIntersecting) {
          entry.target.classList.add('visible');
          revealObserver.unobserve(entry.target);
        }
      });
    }, { threshold: 0.12 })
  : null;

$$('.reveal').forEach((element) => {
  if (revealObserver) revealObserver.observe(element);
  else element.classList.add('visible');
});

// Lightweight neural-network ambience for the hero stage.
const canvas = $('#neuralCanvas');
const context = canvas?.getContext('2d');
let canvasFrame;
let canvasNodes = [];

const resizeCanvas = () => {
  if (!canvas || !context) return;
  const rect = canvas.getBoundingClientRect();
  const ratio = Math.min(window.devicePixelRatio || 1, 2);
  canvas.width = Math.max(1, Math.floor(rect.width * ratio));
  canvas.height = Math.max(1, Math.floor(rect.height * ratio));
  context.setTransform(ratio, 0, 0, ratio, 0, 0);
  const width = rect.width;
  const height = rect.height;
  const columns = Math.max(4, Math.round(width / 125));
  canvasNodes = [];
  for (let column = 0; column < columns; column += 1) {
    const count = 4 + (column % 3);
    for (let row = 0; row < count; row += 1) {
      canvasNodes.push({
        x: width * (0.08 + (column / Math.max(1, columns - 1)) * 0.84) + (Math.random() - 0.5) * 18,
        y: height * (0.18 + (row / Math.max(1, count - 1)) * 0.64) + (Math.random() - 0.5) * 20,
        r: 1.4 + Math.random() * 2,
        phase: Math.random() * Math.PI * 2,
        tint: Math.random() > 0.75 ? 'violet' : Math.random() > 0.55 ? 'peach' : 'cyan',
      });
    }
  }
};

const drawCanvas = (time = 0) => {
  if (!canvas || !context) return;
  const { width, height } = canvas.getBoundingClientRect();
  context.clearRect(0, 0, width, height);
  const motion = body.classList.contains('reduced-motion') ? 0 : time / 1000;
  const colors = { cyan: '121,231,237', violet: '187,155,255', peach: '255,179,138' };

  for (let i = 0; i < canvasNodes.length; i += 1) {
    const a = canvasNodes[i];
    for (let j = i + 1; j < canvasNodes.length; j += 1) {
      const b = canvasNodes[j];
      const dx = a.x - b.x;
      const dy = a.y - b.y;
      const distance = Math.sqrt(dx * dx + dy * dy);
      if (distance < 145) {
        const alpha = (1 - distance / 145) * 0.14;
        context.strokeStyle = `rgba(121,231,237,${alpha})`;
        context.lineWidth = 1;
        context.beginPath();
        context.moveTo(a.x, a.y);
        context.lineTo(b.x, b.y);
        context.stroke();
      }
    }
  }

  canvasNodes.forEach((node) => {
    const shimmer = body.classList.contains('reduced-motion') ? 0.7 : 0.58 + Math.sin(motion * 1.6 + node.phase) * 0.22;
    context.beginPath();
    context.fillStyle = `rgba(${colors[node.tint]},${Math.max(0.25, shimmer)})`;
    context.shadowColor = `rgba(${colors[node.tint]},0.7)`;
    context.shadowBlur = 10;
    context.arc(node.x, node.y, node.r, 0, Math.PI * 2);
    context.fill();
    context.shadowBlur = 0;
  });

  if (!body.classList.contains('reduced-motion')) canvasFrame = requestAnimationFrame(drawCanvas);
};

if (canvas && context) {
  resizeCanvas();
  drawCanvas();
  window.addEventListener('resize', () => {
    resizeCanvas();
    if (body.classList.contains('reduced-motion')) drawCanvas();
  });
}

// Motion toggle gives a presenter a clean, calm mode on demand.
const motionToggle = $('#motionToggle');
const syncMotionLabel = () => {
  const reduced = body.classList.contains('reduced-motion');
  if (motionToggle) {
    motionToggle.setAttribute('aria-pressed', String(reduced));
    $('.button-label', motionToggle).textContent = reduced ? 'Motion off' : 'Motion on';
  }
  const motionStatus = $('#motionStatus');
  if (motionStatus) motionStatus.innerHTML = `<i></i> ${reduced ? 'paused' : 'playing'}`;
};

if (prefersReducedMotion) body.classList.add('reduced-motion');
syncMotionLabel();
motionToggle?.addEventListener('click', () => {
  body.classList.toggle('reduced-motion');
  if (!body.classList.contains('reduced-motion') && canvas && context) {
    cancelAnimationFrame(canvasFrame);
    canvasFrame = requestAnimationFrame(drawCanvas);
  } else if (canvas && context) {
    cancelAnimationFrame(canvasFrame);
    drawCanvas();
  }
  syncMotionLabel();
});

// Story navigation.
$('#startJourney')?.addEventListener('click', () => scrollToId('story'));
$('#scrollCue')?.addEventListener('click', () => scrollToId('story'));
$('#tryDemo')?.addEventListener('click', () => {
  scrollToId('lab');
  window.setTimeout(() => $('#argumentInput')?.focus(), 650);
});
$('#backToTop')?.addEventListener('click', () => window.scrollTo({ top: 0, behavior: body.classList.contains('reduced-motion') ? 'auto' : 'smooth' }));

// Presentation mode is useful when the page is projected in a room.
const presentButton = $('#presentButton');
presentButton?.addEventListener('click', async () => {
  body.classList.toggle('presentation-mode');
  const active = body.classList.contains('presentation-mode');
  presentButton.setAttribute('aria-pressed', String(active));
  presentButton.innerHTML = active ? '<span class="button-icon">×</span> Exit' : '<span class="button-icon">↗</span> Present';
  if (active) {
    toast('Presentation mode on — the story is ready to walk through.');
    try {
      if (!document.fullscreenElement) await document.documentElement.requestFullscreen?.();
    } catch {
      // Fullscreen can be blocked in a preview; the visual mode still works.
    }
  } else if (document.fullscreenElement) {
    await document.exitFullscreen?.();
  }
});

document.addEventListener('fullscreenchange', () => {
  if (!document.fullscreenElement && body.classList.contains('presentation-mode')) {
    body.classList.remove('presentation-mode');
    presentButton?.setAttribute('aria-pressed', 'false');
    if (presentButton) presentButton.innerHTML = '<span class="button-icon">↗</span> Present';
  }
});

// The four stops in the explanatory trace.
const traceStates = [
  {
    type: 'RAW ARGUMENT',
    counter: '01 / 04',
    content: '<div class="argument-line"><span class="line-number">01</span><span>“Everyone in our class agrees,</span></div><div class="argument-line highlighted"><span class="line-number">02</span><span>so this must be the right answer.”</span></div>',
    footer: 'A sentence is still a sentence — before the model sees numbers.',
  },
  {
    type: 'TOKEN REPRESENTATION',
    counter: '02 / 04',
    content: '<div class="argument-line"><span class="line-number">01</span><span><b style="color:var(--cyan)">everyone</b> · <b style="color:var(--peach)">agrees</b> · <b style="color:var(--cyan)">right</b></span></div><div class="argument-line highlighted"><span class="line-number">02</span><span>words become learnable coordinates.</span></div>',
    footer: 'The model does not read “meaning” directly — it works with representations.',
  },
  {
    type: 'STRUCTURE SIGNAL',
    counter: '03 / 04',
    content: '<div class="argument-line"><span class="line-number">01</span><span><b style="color:var(--peach)">group agreement</b> → conclusion</span></div><div class="argument-line highlighted"><span class="line-number">02</span><span>the relationship becomes evidence.</span></div>',
    footer: 'Architecture determines which relationships are easiest to notice.',
  },
  {
    type: 'HUMAN-READABLE HYPOTHESIS',
    counter: '04 / 04',
    content: '<div class="argument-line"><span class="line-number">01</span><span><b style="color:var(--cyan)">possible signal:</b></span></div><div class="argument-line highlighted"><span class="line-number">02</span><span>ad populum / appeal to popularity</span></div>',
    footer: 'A label opens the conversation; it does not close the argument.',
  },
];

const setJourneyStep = (index) => {
  const state = traceStates[index];
  $$('.journey-step').forEach((step, stepIndex) => {
    const active = stepIndex === index;
    step.classList.toggle('active', active);
    step.setAttribute('aria-pressed', String(active));
  });
  $$('.rail-node').forEach((node, nodeIndex) => node.classList.toggle('active', nodeIndex === index));
  $('#traceType').textContent = state.type;
  $('#traceCounter').textContent = state.counter;
  $('#traceContent').innerHTML = state.content;
  $('#traceFooter').textContent = state.footer;
  const card = $('#traceCard');
  playMotion(card, [{ transform: 'translateY(4px)', opacity: .55 }, { transform: 'translateY(0)', opacity: 1 }], { duration: 360, easing: 'ease-out' });
};

$$('.journey-step').forEach((step) => {
  step.addEventListener('mouseenter', () => setJourneyStep(Number(step.dataset.step)));
  step.addEventListener('focusin', () => setJourneyStep(Number(step.dataset.step)));
  step.addEventListener('click', () => setJourneyStep(Number(step.dataset.step)));
  step.addEventListener('keydown', (event) => {
    if (event.key === 'Enter' || event.key === ' ') {
      event.preventDefault();
      setJourneyStep(Number(step.dataset.step));
    }
  });
});

const modelData = {
  cnn: {
    kicker: 'LOCAL PATTERN DETECTOR', index: '01 / 03', title: 'CNN', titleAccent: 'sees nearby phrases.',
    description: 'Sliding filters scan short windows of tokens, then keep the strongest pattern they find. It is fast, compact, and naturally good at phrase-level cues.',
    question: '“Which local phrase is carrying the argument?”',
    flow: ['tokens', 'sliding windows', 'max-pool', 'class signal'],
    note: 'Think of this as a highlighter that keeps the most useful fragments.',
    accent: 'cyan',
  },
  bilstm: {
    kicker: 'SEQUENCE CONTEXT READER', index: '02 / 03', title: 'BiLSTM', titleAccent: 'reads in both directions.',
    description: 'A recurrent memory moves through the sequence from left to right and right to left. The final representation carries context from the whole argument.',
    question: '“How does this phrase change what came before and after it?”',
    flow: ['tokens', 'forward memory', 'backward memory', 'context signal'],
    note: 'Like reading the same sentence twice — once forward, once backward — then comparing notes.',
    accent: 'peach',
  },
  transformer: {
    kicker: 'RELATIONSHIP MAPPER', index: '03 / 03', title: 'Transformer', titleAccent: 'connects distant words.',
    description: 'Attention lets each token look toward other tokens and build a context-aware representation. Distant relationships can become visible in one layer.',
    question: '“Which words should pay attention to each other?”',
    flow: ['tokens', 'attention links', 'context mix', 'class signal'],
    note: 'Imagine every word drawing temporary threads to the words that help explain it.',
    accent: 'violet',
  },
};

const svgShell = (content) => `<svg class="model-svg" viewBox="0 0 620 430" role="img" aria-label="Animated model architecture diagram" xmlns="http://www.w3.org/2000/svg">${content}</svg>`;

const architectureSVG = (name) => {
  if (name === 'bilstm') {
    return svgShell(`
      <defs><linearGradient id="bGrad" x1="0" x2="1"><stop stop-color="#ffb38a" stop-opacity=".12"/><stop offset="1" stop-color="#ffb38a" stop-opacity=".42"/></linearGradient></defs>
      <text x="32" y="44" class="model-label">SEQUENCE IN</text>
      <g transform="translate(34 72)"><rect class="model-node" width="72" height="32" rx="7"/><text x="13" y="20" class="model-label">ignore</text></g>
      <g transform="translate(128 72)"><rect class="model-node peach" width="72" height="32" rx="7"/><text x="14" y="20" class="model-label-bright">her</text></g>
      <g transform="translate(222 72)"><rect class="model-node" width="72" height="32" rx="7"/><text x="11" y="20" class="model-label">proposal</text></g>
      <g transform="translate(316 72)"><rect class="model-node peach" width="72" height="32" rx="7"/><text x="14" y="20" class="model-label-bright">failed</text></g>
      <g transform="translate(410 72)"><rect class="model-node" width="72" height="32" rx="7"/><text x="16" y="20" class="model-label">trust</text></g>
      <path class="model-line peach" d="M70 180 C180 120, 330 120, 500 180"/><path class="model-line peach" d="M500 260 C350 315, 210 315, 70 260"/>
      <g transform="translate(56 163)"><rect class="model-node peach" width="62" height="46" rx="12"/><text x="17" y="28" class="model-label-bright">h₁ →</text></g>
      <g transform="translate(498 163)"><rect class="model-node peach" width="62" height="46" rx="12"/><text x="17" y="28" class="model-label-bright">← hₙ</text></g>
      <rect class="model-scan" x="142" y="151" width="326" height="70" rx="14"/>
      <path class="model-line" d="M118 186 H498"/><circle class="model-flow-dot peach" cx="118" cy="186" r="4"/>
      <path class="model-line peach" d="M118 186 C210 140, 360 140, 498 186"/><path class="model-line peach" d="M118 186 C210 235, 360 235, 498 186"/>
      <g transform="translate(256 282)"><rect class="model-node peach" width="110" height="48" rx="14" fill="url(#bGrad)"/><text x="17" y="21" class="model-label">CONTEXT</text><text x="17" y="36" class="model-label-bright">merge memories</text></g>
      <path class="model-line peach" d="M87 209 C120 250, 190 280, 256 302 M529 209 C495 250, 427 280, 366 302"/>
      <path class="model-line" d="M311 330 V370"/><circle class="model-flow-dot peach" cx="311" cy="370" r="5"/>
      <text x="268" y="397" class="model-label">SIGNAL OUT</text>
    `);
  }
  if (name === 'transformer') {
    return svgShell(`
      <text x="34" y="43" class="model-label">ATTENTION MAP</text>
      <g transform="translate(45 73)"><rect class="model-node violet" width="82" height="34" rx="8"/><text x="15" y="21" class="model-label-bright">ignore</text></g>
      <g transform="translate(169 73)"><rect class="model-node violet" width="82" height="34" rx="8"/><text x="24" y="21" class="model-label-bright">her</text></g>
      <g transform="translate(293 73)"><rect class="model-node violet" width="82" height="34" rx="8"/><text x="15" y="21" class="model-label-bright">proposal</text></g>
      <g transform="translate(417 73)"><rect class="model-node violet" width="82" height="34" rx="8"/><text x="17" y="21" class="model-label-bright">failed</text></g>
      <path class="model-attention" d="M86 106 Q210 270 334 106"/><path class="model-attention" d="M210 106 Q332 270 458 106"/><path class="model-attention" d="M334 106 Q210 300 86 106"/>
      <path class="model-attention" d="M86 106 Q293 205 458 106"/><path class="model-attention" d="M458 106 Q293 246 86 106"/>
      <circle cx="86" cy="106" r="7" fill="#bb9bff" opacity=".85"/><circle cx="210" cy="106" r="7" fill="#bb9bff" opacity=".85"/><circle cx="334" cy="106" r="7" fill="#bb9bff" opacity=".85"/><circle cx="458" cy="106" r="7" fill="#bb9bff" opacity=".85"/>
      <g transform="translate(184 194)"><rect class="model-node violet" width="218" height="70" rx="17"/><text x="19" y="27" class="model-label">CONTEXT MIX</text><text x="19" y="49" class="model-label-bright">every token informs another</text></g>
      <path class="model-line violet" d="M86 113 C128 166, 173 189, 210 194 M210 113 C220 157, 244 180, 260 194 M334 113 C326 157, 300 180, 290 194 M458 113 C420 166, 387 188, 374 194"/>
      <circle class="model-flow-dot violet" cx="86" cy="113" r="4"/><circle class="model-flow-dot violet" cx="458" cy="113" r="4"/>
      <path class="model-line violet" d="M293 264 V346"/><circle class="model-flow-dot violet" cx="293" cy="346" r="5"/>
      <text x="247" y="383" class="model-label">SIGNAL OUT</text>
    `);
  }
  return svgShell(`
    <text x="34" y="43" class="model-label">SLIDING WINDOWS</text>
    <g transform="translate(35 85)"><rect class="model-node" width="72" height="34" rx="8"/><text x="15" y="21" class="model-label">everyone</text></g>
    <g transform="translate(125 85)"><rect class="model-node highlight" width="72" height="34" rx="8"/><text x="15" y="21" class="model-label-bright">agrees</text></g>
    <g transform="translate(215 85)"><rect class="model-node" width="72" height="34" rx="8"/><text x="18" y="21" class="model-label">so</text></g>
    <g transform="translate(305 85)"><rect class="model-node highlight" width="72" height="34" rx="8"/><text x="17" y="21" class="model-label-bright">right</text></g>
    <g transform="translate(395 85)"><rect class="model-node" width="72" height="34" rx="8"/><text x="17" y="21" class="model-label">answer</text></g>
    <rect class="model-scan" x="113" y="64" width="196" height="76" rx="14"/>
    <path class="model-line" d="M71 119 C128 170, 230 170, 300 119"/><path class="model-line" d="M161 119 C218 170, 320 170, 390 119"/>
    <circle class="model-flow-dot" cx="71" cy="119" r="4"/><circle class="model-flow-dot" cx="161" cy="119" r="4"/>
    <g transform="translate(103 213)"><rect class="model-node highlight" width="112" height="50" rx="14"/><text x="17" y="22" class="model-label">FILTER 01</text><text x="17" y="39" class="model-label-bright">group → truth</text></g>
    <g transform="translate(270 213)"><rect class="model-node" width="112" height="50" rx="14"/><text x="17" y="22" class="model-label">FILTER 02</text><text x="17" y="39" class="model-label">so → right</text></g>
    <path class="model-line" d="M145 140 V213 M251 140 V213 M341 140 V213"/>
    <path class="model-line" d="M159 263 C190 304, 250 319, 309 263"/><circle class="model-flow-dot" cx="159" cy="263" r="4"/>
    <g transform="translate(227 310)"><rect class="model-node highlight" width="112" height="48" rx="14"/><text x="18" y="21" class="model-label">MAX-POOL</text><text x="18" y="36" class="model-label-bright">keep strongest</text></g>
    <path class="model-line" d="M283 358 V385"/><circle class="model-flow-dot" cx="283" cy="385" r="5"/><text x="237" y="415" class="model-label">SIGNAL OUT</text>
  `);
};

const renderModel = (name) => {
  const data = modelData[name];
  $('#modelStage').innerHTML = architectureSVG(name);
  $('#modelKicker').textContent = data.kicker;
  $('#modelIndex').textContent = data.index;
  $('#modelTitle').innerHTML = `${data.title} <em>${data.titleAccent}</em>`;
  $('#modelDescription').textContent = data.description;
  $('#modelQuestion').textContent = data.question;
  $('#modelFlow').innerHTML = data.flow.map((item) => `<span class="flow-chip">${item}</span>`).join('');
  $('#modelNote').textContent = data.note;
  $$('.model-tab').forEach((tab) => {
    const selected = tab.dataset.model === name;
    tab.classList.toggle('active', selected);
    tab.setAttribute('aria-pressed', String(selected));
  });
  const info = $('.model-info');
  playMotion(info, [{ opacity: .4, transform: 'translateY(6px)' }, { opacity: 1, transform: 'translateY(0)' }], { duration: 400, easing: 'ease-out' });
};

$$('.model-tab').forEach((tab) => tab.addEventListener('click', () => renderModel(tab.dataset.model)));
renderModel('cnn');

window.addEventListener('keydown', (event) => {
  if (event.target.matches('textarea, input')) return;
  const keyMap = { '1': 'cnn', '2': 'bilstm', '3': 'transformer' };
  if (keyMap[event.key]) {
    renderModel(keyMap[event.key]);
    scrollToId('models');
  }
});

// Interactive presentation lab. These cues are deliberately explanatory,
// not a hidden replacement for the trained Python models.
const demoPatterns = [
  {
    test: /(everyone|everybody|all of us|class agrees|popular|most people|majority)/i,
    label: 'Ad populum',
    reason: 'The argument treats group agreement or popularity as evidence that the conclusion must be correct.',
    signals: ['everyone', 'agrees', 'group agreement', 'conclusion'],
  },
  {
    test: /(stupid|idiot|cannot be trusted|failed last|bad person|who are you|she is|he is)/i,
    label: 'Ad hominem',
    reason: 'The focus shifts from the proposal to a personal attack on the person making it.',
    signals: ['person', 'credibility', 'proposal', 'attack'],
  },
  {
    test: /(if we allow|soon nobody|then everyone|next thing|end of the world|slippery)/i,
    label: 'False causality',
    reason: 'The argument presents a chain of consequences without establishing that the first step causes the next one.',
    signals: ['if', 'then', 'consequence', 'causal leap'],
  },
  {
    test: /(either|or else|only two|one of two|if not .* then)/i,
    label: 'False dilemma',
    reason: 'The wording frames a choice as if only two options exist, leaving no room for alternatives or nuance.',
    signals: ['either / or', 'two options', 'missing alternative'],
  },
  {
    test: /(because .* so|therefore|obviously caused|made .* happen|after .* means)/i,
    label: 'False causality',
    reason: 'The conclusion assumes that sequence or correlation is enough to establish cause and effect.',
    signals: ['because', 'therefore', 'sequence', 'causal leap'],
  },
  {
    test: /(feel|heart|everyone will suffer|think of the children|shame|terrible person)/i,
    label: 'Appeal to emotion',
    reason: 'The argument leans on an emotional reaction instead of giving enough support for the conclusion.',
    signals: ['emotion', 'reaction', 'pressure', 'conclusion'],
  },
];

const fallbackResult = {
  label: 'Signal needs context',
  reason: 'The text contains a claim worth inspecting, but this presentation trace does not force a fallacy label without more context.',
  signals: ['claim', 'context', 'human review'],
};

const tokenizeForDisplay = (text) => text.match(/[A-Za-z0-9]+(?:['’][A-Za-z]+)?|[^\s]/g) || [];

const getIllustrativeResult = (text) => demoPatterns.find((pattern) => pattern.test(text)) || fallbackResult;

let analysisRun = 0;

const animateAnalysis = () => {
  const text = $('#argumentInput').value.trim();
  if (!text) {
    toast('Add an argument first so the lens has something to inspect.');
    $('#argumentInput').focus();
    return;
  }

  analysisRun += 1;
  const run = analysisRun;
  const result = getIllustrativeResult(text);
  const tokens = tokenizeForDisplay(text);
  const signalWords = new Set(result.signals.flatMap((signal) => signal.toLowerCase().split(/[^a-z]+/).filter(Boolean)));
  const placeholder = $('#analysisPlaceholder');
  const running = $('#analysisRunning');
  const output = $('#analysisResult');
  const stream = $('#tokenStream');
  const progress = $('#analysisProgress');
  const runningLabel = $('#runningLabel');

  placeholder.classList.add('hidden');
  output.classList.add('hidden');
  running.classList.remove('hidden');
  stream.innerHTML = '';
  progress.style.width = '0%';
  $('#analysisStatus').innerHTML = '<i></i> tracing the signal';

  const phases = [
    { at: 0, label: 'Splitting the argument into tokens', width: 16 },
    { at: 520, label: 'Mapping words into a representation', width: 38 },
    { at: 1050, label: 'Passing the sequence through a model lens', width: 64 },
    { at: 1600, label: 'Surfacing the strongest explainable signal', width: 88 },
  ];

  phases.forEach((phase) => {
    window.setTimeout(() => {
      if (run !== analysisRun) return;
      runningLabel.textContent = phase.label;
      progress.style.width = `${phase.width}%`;
      if (phase.width === 16) {
        tokens.forEach((token, index) => {
          const chip = document.createElement('span');
          chip.className = `token${signalWords.has(token.toLowerCase().replace(/[^a-z0-9]/g, '')) ? ' signal' : ''}`;
          chip.textContent = token;
          chip.style.animationDelay = `${index * 45}ms`;
          stream.appendChild(chip);
        });
      }
    }, phase.at);
  });

  window.setTimeout(() => {
    if (run !== analysisRun) return;
    running.classList.add('hidden');
    output.classList.remove('hidden');
    $('#resultLabel').textContent = result.label;
    $('#resultReason').textContent = result.reason;
    $('#signalList').innerHTML = result.signals.map((signal) => `<span class="signal-chip">${signal}</span>`).join('');
    $('#analysisStatus').innerHTML = '<i></i> signal surfaced';
    playMotion(output, [{ opacity: 0, transform: 'translateY(12px)' }, { opacity: 1, transform: 'translateY(0)' }], { duration: 520, easing: 'cubic-bezier(.2,.8,.2,1)' });
  }, 2250);
};

$('#analyzeButton')?.addEventListener('click', animateAnalysis);
$('#replayButton')?.addEventListener('click', animateAnalysis);
$('#argumentInput')?.addEventListener('keydown', (event) => {
  if ((event.metaKey || event.ctrlKey) && event.key === 'Enter') animateAnalysis();
});

$$('.example-chip').forEach((chip) => chip.addEventListener('click', () => {
  $('#argumentInput').value = chip.dataset.example;
  $('#argumentInput').focus();
  toast('Example loaded — animate the trace when ready.');
}));

$('#clearButton')?.addEventListener('click', () => {
  analysisRun += 1;
  $('#argumentInput').value = '';
  $('#analysisPlaceholder').classList.remove('hidden');
  $('#analysisRunning').classList.add('hidden');
  $('#analysisResult').classList.add('hidden');
  $('#analysisStatus').innerHTML = '<i></i> waiting for input';
  $('#argumentInput').focus();
});

// A small cue if the page is opened directly from the filesystem. Browsers
// still render the page, but this tells the presenter the preferred command.
if (window.location.protocol === 'file:') {
  window.setTimeout(() => toast('Tip: run `python -m http.server 8000 -d visualizer` for the smoothest preview.'), 1300);
}
