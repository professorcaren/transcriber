import { toSegments, assignSpeakers } from './diar.js';
import * as X from './export.js';

const $ = id => document.getElementById(id);
const COLORS = [...Array(8)].map((_, i) => getComputedStyle(document.documentElement).getPropertyValue('--s' + i).trim());

// Whisper presets, using the prebuilt dtype variants in the onnx-community repos.
// `_timestamped` exports carry the cross-attentions needed for word timings.
// On GPU: fp16 base matches fp32 on 99% of words; small and turbo use q4f16 decoders.
const QUALITY = {
  fast: {
    label: 'Whisper base', id: 'onnx-community/whisper-base_timestamped',
    webgpu: { dtype: { encoder_model: 'fp16', decoder_model_merged: 'fp16' }, mb: 146 },
    wasm: { dtype: 'q8', mb: 77 },
  },
  better: {
    label: 'Whisper small', id: 'onnx-community/whisper-small_timestamped',
    webgpu: { dtype: { encoder_model: 'fp16', decoder_model_merged: 'q4f16' }, mb: 322 },
    wasm: { dtype: 'q8', mb: 249 },
  },
  best: {
    label: 'Whisper large-v3-turbo', id: 'onnx-community/whisper-large-v3-turbo_timestamped',
    webgpu: { dtype: { encoder_model: 'q4f16', decoder_model_merged: 'q4f16' }, mb: 564 },
    wasm: null, // far too slow on CPU
  },
};
const DIAR_LABEL = 'NVIDIA Nemotron 3 Diarization (int8)';
// ONNX export of the diarizer; `?models=local` serves ./models/ instead (for development)
const DIAR_BASE = new URLSearchParams(location.search).get('models') === 'local'
  ? new URL('models/', location.href).href
  : 'https://huggingface.co/NealCaren/Nemotron-3-Diarization-ONNX/resolve/main/';

let gpu = false;               // WebGPU usable for Whisper
let file = null, audio16 = null;
let diar = null, words = null, segs = [], turns = [];
let job = null, dirty = false, lastModelLabel = null;
const names = [...Array(8)].map((_, i) => `Speaker ${i + 1}`);

const diarWorker = new Worker('./worker.js', { type: 'module' });
const asrWorker = new Worker('./whisper-worker.js', { type: 'module' });

// ---------- setup ----------
(async () => {
  try {
    const adapter = await navigator.gpu?.requestAdapter();
    gpu = !!adapter && adapter.features.has('shader-f16');
  } catch { gpu = false; }
  $('device').value = 'auto';
  updateOptions();
})();

function asrDevice() {
  const d = $('device').value;
  return d === 'auto' ? (gpu ? 'webgpu' : 'wasm') : d;
}

function updateOptions() {
  const dev = asrDevice();
  for (const [key, q] of Object.entries(QUALITY)) {
    const input = document.querySelector(`input[name=quality][value=${key}]`);
    const spec = q[dev];
    input.disabled = !spec;
    input.closest('label').classList.toggle('disabled', !spec);
    $('mb-' + key).textContent = spec ? `${spec.mb} MB download` : 'needs a GPU';
  }
  if (document.querySelector('input[name=quality]:checked').disabled) document.querySelector('input[name=quality][value=fast]').checked = true;
  $('device-note').textContent = dev === 'webgpu' ? 'Using your graphics card (WebGPU).' : 'Using your processor (no WebGPU available, or CPU selected).';
  $('quality-box').classList.toggle('off', !$('opt-asr').checked);
  $('go').disabled = !audio16 || (!$('opt-asr').checked && !$('opt-diar').checked) || !!job;
}
['device', 'opt-asr', 'opt-diar'].forEach(id => $(id).addEventListener('change', updateOptions));
document.querySelectorAll('input[name=quality]').forEach(el => el.addEventListener('change', updateOptions));

// ---------- audio input ----------
async function loadAudio(blob, name) {
  if (dirty && !confirmDiscard()) return;
  $('file-info').textContent = `Reading ${name}…`;
  try {
    const decoded = await new AudioContext().decodeAudioData(await blob.arrayBuffer());
    const off = new OfflineAudioContext(1, Math.ceil(decoded.duration * 16000), 16000);
    const src = off.createBufferSource(); src.buffer = decoded; src.connect(off.destination); src.start();
    audio16 = (await off.startRendering()).getChannelData(0);
    if (file?.url) URL.revokeObjectURL(file.url);
    file = { name, duration: decoded.duration, url: URL.createObjectURL(blob) };
  } catch (e) {
    $('file-info').textContent = `Couldn't read ${name}. Try an mp3, wav or m4a file.`;
    return;
  }
  $('player').src = file.url;
  $('drop').classList.add('loaded');
  $('file-info').innerHTML = `<b>${esc(name)}</b> · ${X.hms(file.duration)} long`;
  diar = words = null; turns = []; dirty = false;
  $('results').hidden = true;
  $('progress').hidden = true;
  updateOptions();
}

function confirmDiscard() {
  // no confirm() dialogs: ask inline by requiring a second click
  if (confirmDiscard.armed) { confirmDiscard.armed = false; return true; }
  confirmDiscard.armed = true;
  $('file-info').textContent = 'You have an unsaved transcript. Export it first, or choose the file again to discard it.';
  return false;
}

$('pick').onclick = () => $('file').click();
$('file').onchange = e => { const f = e.target.files[0]; e.target.value = ''; if (f) loadAudio(f, f.name); };
$('sample').onclick = async e => { e.preventDefault(); loadAudio(await (await fetch('sample.wav')).blob(), 'sample-meeting.wav'); };
const drop = $('drop');
drop.ondragover = e => { e.preventDefault(); drop.classList.add('over'); };
drop.ondragleave = () => drop.classList.remove('over');
drop.ondrop = e => { e.preventDefault(); drop.classList.remove('over'); const f = e.dataTransfer.files[0]; if (f) loadAudio(f, f.name); };

let recorder = null, recStart = 0, recTimer = null;
$('rec').onclick = async () => {
  if (recorder) { recorder.stop(); return; }
  let stream;
  try { stream = await navigator.mediaDevices.getUserMedia({ audio: true }); }
  catch { $('file-info').textContent = 'Microphone access was blocked.'; return; }
  const chunks = [];
  recorder = new MediaRecorder(stream);
  recorder.ondataavailable = e => chunks.push(e.data);
  recorder.onstop = () => {
    stream.getTracks().forEach(t => t.stop());
    clearInterval(recTimer); recorder = null; $('rec').textContent = 'Record';
    const stamp = new Date().toISOString().slice(0, 16).replace('T', ' ').replace(':', '-');
    loadAudio(new Blob(chunks, { type: chunks[0]?.type }), `recording ${stamp}.webm`);
  };
  recorder.start(); recStart = Date.now();
  recTimer = setInterval(() => { $('rec').textContent = `Stop (${X.hms((Date.now() - recStart) / 1000)})`; }, 500);
};

// ---------- run ----------
$('go').onclick = () => {
  const doAsr = $('opt-asr').checked, doDiar = $('opt-diar').checked;
  const q = QUALITY[document.querySelector('input[name=quality]:checked').value];
  const dev = asrDevice();
  job = { doAsr, doDiar, q, dev, pending: 0, t0: performance.now() };
  diar = words = null; turns = []; segs = []; dirty = false;
  $('transcript').innerHTML = ''; $('results').hidden = true; $('exports').hidden = true;
  $('progress').hidden = false;
  $('row-diar').hidden = !doDiar; $('row-asr').hidden = !doAsr;
  updateOptions();
  if (doDiar) {
    job.pending++;
    const a = audio16.slice();
    diarWorker.postMessage({ type: 'run', audio: a, base: DIAR_BASE, model: 'step_int8', backend: 'wasm' }, [a.buffer]);
    st('diar', 'Starting…', 0);
  }
  if (doAsr) {
    job.pending++;
    const b = audio16.slice();
    asrWorker.postMessage({ audio: b, model: q.id, device: dev, dtype: q[dev].dtype, language: $('lang').value }, [b.buffer]);
    st('asr', 'Starting…', 0);
  }
};

function st(which, text, p) {
  $('st-' + which).textContent = text;
  $('pr-' + which).style.width = p == null ? '0' : (100 * p).toFixed(1) + '%';
}

function finishOne() {
  if (--job.pending > 0) return;
  const secs = (performance.now() - job.t0) / 1000;
  $('done-note').textContent = `Finished in ${secs < 90 ? Math.round(secs) + ' seconds' : Math.round(secs / 60) + ' minutes'}.`;
  buildTurns(true);
  job.finished = true;
  lastModelLabel = job.doAsr ? job.q.label : null;
  renderTranscript();
  const hasText = !!(words && words.length);
  document.querySelectorAll('#exports button[data-f]').forEach(b => {
    const f = b.dataset.f;
    b.disabled = (f === 'rttm' && !diar) || (['srt', 'vtt'].includes(f) && !hasText);
  });
  $('exports').hidden = false;
  job = null; dirty = true;
  updateOptions();
}

diarWorker.onmessage = ({ data }) => {
  if (data.type === 'status') st('diar', data.text.replace('Diarizing…', 'Finding speakers…'), data.progress);
  else if (data.type === 'loaded') st('diar', 'Finding speakers…', 0);
  else if (data.type === 'error') { st('diar', 'Something went wrong: ' + data.text); finishOne(); }
  else if (data.type === 'result') {
    diar = data;
    segs = toSegments(diar.probs, diar.numFrames, 0.5, 0.2);
    st('diar', `Done: found ${new Set(segs.map(s => s.speaker)).size} speaker(s).`, 1);
    showResults(); finishOne();
  }
};

asrWorker.onmessage = ({ data }) => {
  if (data.type === 'status') {
    if (data.text.startsWith('Transcribing')) job.asrT0 = performance.now();
    st('asr', data.text.replace('Downloading Whisper', 'Downloading the speech model (first time only)'), data.progress);
  }
  else if (data.type === 'error') { st('asr', 'Something went wrong: ' + data.text); finishOne(); }
  else if (data.type === 'partial') {
    words = data.words;
    // estimate from transcription time only (not the model download)
    const el = (performance.now() - (job.asrT0 ?? job.t0)) / 1000;
    const eta = data.progress > 0.02 ? el / data.progress - el : null;
    st('asr', `Transcribing… ${Math.round(100 * data.progress)}%` + (eta ? `, about ${eta < 90 ? Math.round(eta) + ' s' : Math.round(eta / 60) + ' min'} left` : ''), data.progress);
    showResults();
  } else if (data.type === 'result') {
    words = data.words;
    st('asr', `Done: ${words.length.toLocaleString()} words.`, 1);
    showResults(); finishOne();
  }
};

// ---------- transcript model ----------
const endsSentence = w => /[.?!]["')\]]?$/.test(w.text.trim());

// split a run of words into paragraphs at sentence ends after a pause or ~120 words
function paragraphs(ws) {
  const paras = []; let cur = [];
  ws.forEach((w, i) => {
    cur.push(w);
    const next = ws[i + 1];
    if (next && endsSentence(w) && (next.start - w.end > 1.5 || cur.length >= 120)) { paras.push(cur); cur = []; }
  });
  if (cur.length) paras.push(cur);
  return paras;
}
const joinWords = ws => ws.map(w => w.text).join('').trim();

function buildTurns(final = false) {
  if (words && words.length) {
    const labelled = diar ? assignSpeakers(words, diar.probs, diar.numFrames) : words.map(w => ({ ...w, speaker: null }));
    turns = [];
    if (diar) {
      for (const w of labelled) {
        const last = turns[turns.length - 1];
        if (last && last.speaker === w.speaker) last.words.push(w);
        else turns.push({ speaker: w.speaker, words: [w] });
      }
      for (const t of turns) t.text = paragraphs(t.words).map(joinWords).join('\n\n');
    } else {
      // no speakers: one block per paragraph
      turns = paragraphs(labelled).map(p => ({ speaker: null, words: p, text: joinWords(p) }));
    }
    turns.forEach(t => { t.start = t.words[0].start; t.end = t.words[t.words.length - 1].end; });
  } else if (diar && !(job?.doAsr && !final)) {
    turns = [];
    for (const s of segs) {
      const last = turns[turns.length - 1];
      if (last && last.speaker === s.speaker && s.start - last.end < 1) last.end = s.end;
      else turns.push({ speaker: s.speaker, start: s.start, end: s.end, text: '' });
    }
  } else turns = [];
}

// ---------- rendering ----------
const esc = s => String(s).replace(/[&<>"]/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' })[c]);

function showResults() {
  $('results').hidden = false;
  $('speakers-box').hidden = !diar;
  if (diar) renderSpeakers();
  buildTurns();
  renderTranscript();
  draw();
}

function speakerIds() {
  const talk = new Map();
  for (const s of segs) talk.set(s.speaker, (talk.get(s.speaker) || 0) + s.end - s.start);
  return [...talk.keys()].sort((a, b) => a - b).map(id => ({ id, talk: talk.get(id) }));
}

function renderSpeakers() {
  $('speakers').innerHTML = speakerIds().map(({ id, talk }) =>
    `<label class="spk"><span class="sw" style="background:${COLORS[id]}"></span>` +
    `<input data-s="${id}" value="${esc(names[id])}" aria-label="Name for speaker ${id + 1}">` +
    `<span class="muted">${X.hms(talk).replace(/^00:/, '')} talking</span></label>`).join('');
}
$('speakers').oninput = e => {
  const s = e.target.dataset.s; if (s == null) return;
  names[s] = e.target.value.trim() || `Speaker ${+s + 1}`;
  document.querySelectorAll(`select.who option[value="${s}"]`).forEach(o => o.textContent = names[s]);
  dirty = true;
};

function renderTranscript() {
  const final = !job || job.finished;
  const ids = speakerIds().map(x => x.id);
  if (!turns.length) {
    $('transcript').innerHTML = `<p class="muted">${job ? 'The transcript will appear here as it is written…' : ''}</p>`;
    return;
  }
  $('transcript').innerHTML = turns.map((t, i) => {
    const who = diar && t.speaker != null
      ? `<select class="who" data-i="${i}" style="color:${COLORS[t.speaker]}" ${final ? '' : 'disabled'}>` +
        ids.map(s => `<option value="${s}" ${s === t.speaker ? 'selected' : ''}>${esc(names[s])}</option>`).join('') + `</select>`
      : '';
    const body = t.text
      ? `<div class="txt" data-i="${i}" ${final ? 'contenteditable="plaintext-only" spellcheck="true"' : ''}>${esc(t.text)}</div>`
      : `<div class="txt muted">${job?.doAsr ? '…' : `until ${X.hms(t.end)}`}</div>`;
    return `<div class="turn" data-i="${i}"><button class="ts" data-t="${t.start}" title="Play from here">${X.hms(t.start)}</button><div>${who}${body}</div></div>`;
  }).join('');
  if (!final) $('transcript').lastElementChild?.scrollIntoView({ block: 'nearest' });
}

$('transcript').addEventListener('change', e => {
  if (!e.target.matches('select.who')) return;
  const t = turns[+e.target.dataset.i];
  t.speaker = +e.target.value; e.target.style.color = COLORS[t.speaker]; dirty = true;
});
$('transcript').addEventListener('input', e => {
  if (!e.target.matches('.txt[data-i]')) return;
  const t = turns[+e.target.dataset.i];
  t.text = e.target.innerText.trim(); t.edited = true; dirty = true;
});
$('transcript').addEventListener('click', e => {
  const ts = e.target.closest('.ts'); if (!ts) return;
  $('player').currentTime = +ts.dataset.t; $('player').play();
});

// Esc toggles playback, even while editing
addEventListener('keydown', e => {
  if (e.key !== 'Escape' || !file) return;
  e.preventDefault();
  const p = $('player'); p.paused ? p.play() : p.pause();
});

let nowIdx = -1;
function highlight() {
  const t = $('player').currentTime;
  const i = turns.findIndex(x => t >= x.start && t < x.end + 0.5);
  if (i === nowIdx) return;
  document.querySelector('.turn.now')?.classList.remove('now');
  nowIdx = i;
  const el = document.querySelector(`.turn[data-i="${i}"]`);
  if (!el) return;
  el.classList.add('now');
  const editing = document.activeElement?.classList.contains('txt');
  if ($('follow').checked && !$('player').paused && !editing) el.scrollIntoView({ block: 'center', behavior: 'smooth' });
}

const LANE = 22, TOP = 18;
function draw() {
  const cv = $('timeline');
  cv.hidden = !diar;
  if (!diar || !file) return;
  const ids = speakerIds().map(x => x.id);
  const dpr = devicePixelRatio || 1, W = cv.clientWidth, H = TOP + Math.max(1, ids.length) * LANE;
  cv.width = W * dpr; cv.height = H * dpr; cv.style.height = H + 'px';
  const g = cv.getContext('2d'); g.scale(dpr, dpr);
  const css = getComputedStyle(document.documentElement);
  const dur = file.duration;
  g.font = '11px system-ui'; g.fillStyle = css.getPropertyValue('--muted');
  const tick = [5, 15, 60, 300, 600].find(t => dur / t <= 12) || 1200;
  for (let t = 0; t < dur; t += tick) g.fillText(X.hms(t).replace(/^00:/, ''), t / dur * W + 2, 11);
  ids.forEach((s, lane) => {
    const y = TOP + lane * LANE;
    g.fillStyle = css.getPropertyValue('--line'); g.fillRect(0, y, W, LANE - 6);
    g.fillStyle = COLORS[s];
    for (const seg of segs) if (seg.speaker === s) g.fillRect(seg.start / dur * W, y, Math.max(1, (seg.end - seg.start) / dur * W), LANE - 6);
  });
  g.fillStyle = css.getPropertyValue('--ink');
  g.fillRect($('player').currentTime / dur * W, TOP - 4, 1.5, H - TOP + 4);
}
$('timeline').onclick = e => {
  const r = e.target.getBoundingClientRect();
  $('player').currentTime = (e.clientX - r.left) / r.width * file.duration;
};
$('player').ontimeupdate = () => { draw(); highlight(); };
addEventListener('resize', draw);

// ---------- export ----------
function docModel() {
  return {
    file: file.name, duration: file.duration, created: new Date(),
    diarized: !!diar, transcribed: !!(words && words.length),
    models: { asr: words?.length ? lastModelLabel : null, diar: diar ? DIAR_LABEL : null },
    names, turns, segs,
  };
}

const FORMATS = {
  docx: [d => X.toDocx(d), 'docx'],
  txt: [d => new Blob([X.toTxt(d)], { type: 'text/plain;charset=utf-8' }), 'txt'],
  srt: [d => new Blob([X.toSrt(d)], { type: 'application/x-subrip' }), 'srt'],
  vtt: [d => new Blob([X.toVtt(d)], { type: 'text/vtt' }), 'vtt'],
  csv: [d => new Blob([X.toCsv(d)], { type: 'text/csv;charset=utf-8' }), 'csv'],
  json: [d => new Blob([X.toJson(d)], { type: 'application/json' }), 'json'],
  rttm: [d => new Blob([X.toRttm(d)], { type: 'text/plain' }), 'rttm'],
};
$('exports').addEventListener('click', e => {
  const b = e.target.closest('button[data-f]'); if (!b) return;
  const d = docModel();
  const [make, ext] = FORMATS[b.dataset.f];
  const a = document.createElement('a');
  a.href = URL.createObjectURL(make(d));
  a.download = file.name.replace(/\.[^.]+$/, '') + (b.dataset.f === 'json' ? '.transcript.json' : '.' + ext);
  a.click();
  setTimeout(() => URL.revokeObjectURL(a.href), 5000);
  dirty = false;
});

addEventListener('beforeunload', e => { if (dirty || job) { e.preventDefault(); e.returnValue = ''; } });
