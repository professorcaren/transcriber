// Nemotron 3 Diarization in plain JS + onnxruntime (web or node).
// Port of transformers' Nemotron3DiarizationForAudioFrameClassification (offline mode)
// and NemotronAsrStreamingFeatureExtractor.

export const CFG = {
  sr: 16000, nFft: 512, win: 400, hop: 160, nMels: 128, preemph: 0.97,
  sub: 8, hidden: 512, nSpk: 8,
  chunkLen: 340, rightCtx: 40, fifoLen: 40, updatePeriod: 300,
  cacheLen: 264, silenceFrames: 1, predThresh: 0.25, latestBoost: 0.05,
  minPosRate: 0.5, strongRate: 0.75, weakRate: 1.5,
};

// ---------- features ----------
function makeFFT(n) {
  const levels = Math.log2(n);
  const rev = new Uint32Array(n);
  for (let i = 0; i < n; i++) {
    let r = 0;
    for (let b = 0; b < levels; b++) r |= ((i >> b) & 1) << (levels - 1 - b);
    rev[i] = r;
  }
  const cos = new Float64Array(n / 2), sin = new Float64Array(n / 2);
  for (let i = 0; i < n / 2; i++) { cos[i] = Math.cos(2 * Math.PI * i / n); sin[i] = Math.sin(2 * Math.PI * i / n); }
  return (re, im) => {
    for (let i = 0; i < n; i++) {
      const j = rev[i];
      if (j > i) { let t = re[i]; re[i] = re[j]; re[j] = t; t = im[i]; im[i] = im[j]; im[j] = t; }
    }
    for (let size = 2; size <= n; size *= 2) {
      const half = size / 2, step = n / size;
      for (let i = 0; i < n; i += size) {
        for (let j = 0, k = 0; j < half; j++, k += step) {
          const a = i + j, b = a + half;
          const tr = re[b] * cos[k] + im[b] * sin[k];
          const ti = -re[b] * sin[k] + im[b] * cos[k];
          re[b] = re[a] - tr; im[b] = im[a] - ti;
          re[a] += tr; im[a] += ti;
        }
      }
    }
  };
}

// audio: Float32Array at 16 kHz mono. melFilters: Float32Array [nMels * (nFft/2+1)].
// Returns { data: Float32Array [numFrames * nMels], numFrames } (valid frames only).
export function logMel(audio, melFilters) {
  const { nFft, win, hop, nMels, preemph } = CFG;
  const L = audio.length, nBins = nFft / 2 + 1, pad = nFft / 2;
  const x = new Float32Array(L + 2 * pad);
  x[pad] = audio[0];
  for (let i = 1; i < L; i++) x[pad + i] = audio[i] - preemph * audio[i - 1];
  // symmetric hann(400), zero-padded to 512 and centered, as torch.stft does
  const window = new Float64Array(nFft), off = (nFft - win) / 2;
  for (let i = 0; i < win; i++) window[off + i] = 0.5 - 0.5 * Math.cos(2 * Math.PI * i / (win - 1));
  const numFrames = Math.floor(L / hop);
  const fft = makeFFT(nFft);
  const re = new Float64Array(nFft), im = new Float64Array(nFft), pow = new Float64Array(nBins);
  const out = new Float32Array(numFrames * nMels);
  const guard = 2 ** -24;
  // mel filters are sparse: remember each band's nonzero bin range
  const lo = new Int32Array(nMels), hi = new Int32Array(nMels);
  for (let m = 0; m < nMels; m++) {
    lo[m] = nBins; hi[m] = 0;
    for (let k = 0; k < nBins; k++) if (melFilters[m * nBins + k] !== 0) { lo[m] = Math.min(lo[m], k); hi[m] = k + 1; }
  }
  for (let f = 0; f < numFrames; f++) {
    const s = f * hop;
    for (let i = 0; i < nFft; i++) { re[i] = x[s + i] * window[i]; im[i] = 0; }
    fft(re, im);
    for (let k = 0; k < nBins; k++) pow[k] = re[k] * re[k] + im[k] * im[k];
    for (let m = 0; m < nMels; m++) {
      let acc = 0; const row = m * nBins;
      for (let k = lo[m]; k < hi[m]; k++) acc += melFilters[row + k] * pow[k];
      out[f * nMels + m] = Math.log(acc + guard);
    }
  }
  return { data: out, numFrames };
}

// ---------- speaker cache (batch size 1) ----------
const LOG_HALF = Math.log(0.5);

class SpeakerCache {
  constructor(silence) {
    const c = CFG;
    this.silence = silence;
    this.embeds = [];   // array of Float32Array(512)
    this.probs = [];    // array of Float32Array(8)
    this.fifo = [];
    this.isCompressed = false;
    const budget = Math.floor(c.cacheLen / c.nSpk) - c.silenceFrames;
    this.minPos = Math.floor(budget * c.minPosRate);
    this.nStrong = Math.floor(budget * c.strongRate);
    this.nWeak = Math.floor(budget * c.weakRate);
  }

  getEmbeds() { return this.embeds.concat(this.fifo); }

  // inputEmbeds: array of frames fed to the encoder, probs: pooled probs per input frame
  update(inputEmbeds, probs, numChunkFrames) {
    const c = CFG;
    const nc = this.embeds.length, nf = this.fifo.length;
    const chunkStart = nc + nf;
    let fifoEmb = this.fifo.concat(inputEmbeds.slice(chunkStart, chunkStart + numChunkFrames));
    const Lf = fifoEmb.length;
    let popped = 0;
    if (Lf > c.fifoLen) popped = Math.min(Math.max(c.updatePeriod, Lf - c.fifoLen), Lf);
    if (popped) {
      const fifoProbs = probs.slice(nc, nc + Lf);
      const stored = this.isCompressed ? this.probs : probs.slice(0, nc);
      let cacheE = this.embeds.concat(fifoEmb.slice(0, popped));
      let cacheP = stored.concat(fifoProbs.slice(0, popped));
      fifoEmb = fifoEmb.slice(popped);
      if (cacheE.length > c.cacheLen) {
        [cacheE, cacheP] = this.compress(cacheE, cacheP);
        this.isCompressed = true;
      }
      this.embeds = cacheE; this.probs = cacheP;
    }
    this.fifo = fifoEmb;
  }

  frameScores(probs) {
    const c = CFG, N = probs.length, S = c.nSpk, th = c.predThresh;
    const scores = probs.map(() => new Float64Array(S));
    const posCount = new Int32Array(S);
    for (let t = 0; t < N; t++) {
      const p = probs[t];
      let sumLc = 0; const lc = new Float64Array(S);
      for (let s = 0; s < S; s++) { lc[s] = Math.log(Math.max(1 - p[s], th)); sumLc += lc[s]; }
      for (let s = 0; s < S; s++) {
        let v = Math.log(Math.max(p[s], th)) - lc[s] + sumLc - LOG_HALF;
        if (!(p[s] > 0.5)) v = -Infinity;
        scores[t][s] = v;
        if (v > 0) posCount[s]++;
      }
    }
    for (let t = 0; t < N; t++) for (let s = 0; s < S; s++) {
      const v = scores[t][s];
      if (!(v > 0) && probs[t][s] > 0.5 && posCount[s] >= this.minPos) scores[t][s] = -Infinity;
    }
    return scores;
  }

  boost(scores, k, amount) {
    const N = scores.length;
    for (let s = 0; s < CFG.nSpk; s++) {
      const idx = Array.from({ length: N }, (_, t) => t);
      idx.sort((a, b) => scores[b][s] - scores[a][s] || a - b);
      for (let i = 0; i < Math.min(k, N); i++) scores[idx[i]][s] += amount;
    }
  }

  compress(embeds, probs) {
    const c = CFG, N = probs.length, S = c.nSpk;
    const scores = this.frameScores(probs);
    for (let t = c.cacheLen; t < N; t++) for (let s = 0; s < S; s++) scores[t][s] += c.latestBoost;
    this.boost(scores, this.nStrong, -2 * LOG_HALF);
    this.boost(scores, this.nWeak, -LOG_HALF);
    const nScored = N + c.silenceFrames;
    // flat index = s * nScored + t; silence rows score +inf
    const flat = [];
    for (let s = 0; s < S; s++) for (let t = 0; t < nScored; t++)
      flat.push({ i: s * nScored + t, v: t < N ? scores[t][s] : Infinity });
    flat.sort((a, b) => b.v - a.v || a.i - b.i);
    const sentinel = nScored * S;
    const picked = flat.slice(0, c.cacheLen).map(e => (e.v === -Infinity ? sentinel : e.i)).sort((a, b) => a - b);
    const zero = new Float32Array(S);
    const outE = [], outP = [];
    for (const i of picked) {
      const f = i === sentinel ? N : Math.min(i % nScored, N);
      outE.push(f === N ? this.silence : embeds[f]);
      outP.push(f === N ? zero : probs[f]);
    }
    return [outE, outP];
  }
}

// ---------- model ----------
export class Diarizer {
  // ort: onnxruntime module; embedSession/stepSession: ort.InferenceSession; silence: Float32Array(512)
  constructor(ort, embedSession, stepSession, melFilters, silence) {
    Object.assign(this, { ort, embedSession, stepSession, melFilters, silence });
  }

  async embed(feats, numFrames) {
    const { ort } = this, c = CFG;
    const out = [];
    const block = c.sub * 2048;
    for (let start = 0; start < numFrames; start += block) {
      const n = Math.min(block, numFrames - start);
      const t = new ort.Tensor('float32', feats.subarray(start * c.nMels, (start + n) * c.nMels), [1, n, c.nMels]);
      const r = await this.embedSession.run({ features: t });
      const e = r.embeds.data, T = r.embeds.dims[1];
      for (let i = 0; i < T; i++) out.push(e.slice(i * c.hidden, (i + 1) * c.hidden));
    }
    return out;
  }

  // Returns Float32Array [numFrames * 8] of speaker probabilities (sigmoid), one row per 10 ms.
  async run(audio, onProgress = () => {}) {
    const { ort } = this, c = CFG;
    const { data: feats, numFrames } = logMel(audio, this.melFilters);
    const embeds = await this.embed(feats, numFrames);
    const nEmb = embeds.length;
    const cache = new SpeakerCache(this.silence);
    const probsOut = new Float32Array(Math.ceil(numFrames / c.sub) * c.sub * c.nSpk);
    let written = 0;
    for (let start = 0; start < nEmb; start += c.chunkLen) {
      const end = Math.min(start + c.chunkLen, nEmb);
      const nChunk = end - start;
      const chunk = embeds.slice(start, Math.min(end + c.rightCtx, nEmb));
      const cached = cache.getEmbeds();
      const input = cached.concat(chunk);
      const T = input.length;
      const buf = new Float32Array(T * c.hidden);
      input.forEach((e, i) => buf.set(e, i * c.hidden));
      const r = await this.stepSession.run({ embeds: new ort.Tensor('float32', buf, [1, T, c.hidden]) });
      const logits = r.logits.data; // [T*8, 8]
      // sigmoid + average-pool to encoder rate
      const pooled = [];
      for (let t = 0; t < T; t++) {
        const p = new Float32Array(c.nSpk);
        for (let k = 0; k < c.sub; k++) for (let s = 0; s < c.nSpk; s++)
          p[s] += 1 / (1 + Math.exp(-logits[((t * c.sub + k) * c.nSpk) + s]));
        for (let s = 0; s < c.nSpk; s++) p[s] /= c.sub;
        pooled.push(p);
      }
      cache.update(input, pooled, nChunk);
      const a = cached.length * c.sub * c.nSpk, b = (cached.length + nChunk) * c.sub * c.nSpk;
      for (let i = a; i < b; i++) probsOut[written++] = 1 / (1 + Math.exp(-logits[i]));
      onProgress(end / nEmb);
    }
    return { probs: probsOut.subarray(0, numFrames * c.nSpk), numFrames };
  }
}

// Speaker probabilities -> segments [{start, end, speaker}] in seconds.
export function toSegments(probs, numFrames, threshold = 0.5, minDur = 0) {
  const S = CFG.nSpk, dt = CFG.hop / CFG.sr, segs = [];
  for (let s = 0; s < S; s++) {
    let on = -1;
    for (let t = 0; t <= numFrames; t++) {
      const active = t < numFrames && probs[t * S + s] > threshold;
      if (active && on < 0) on = t;
      if (!active && on >= 0) {
        if ((t - on) * dt >= minDur) segs.push({ start: +(on * dt).toFixed(2), end: +(t * dt).toFixed(2), speaker: s });
        on = -1;
      }
    }
  }
  return segs.sort((a, b) => a.start - b.start || a.speaker - b.speaker);
}

// ---------- transcript alignment ----------
// words: [{text, start, end}] (seconds). Returns words with .speaker, using the summed
// speaker probabilities over each word's span (widened when the span is silent).
export function assignSpeakers(words, probs, numFrames) {
  const S = CFG.nSpk, fps = CFG.sr / CFG.hop;
  const score = (a, b) => {
    const f0 = Math.max(0, Math.floor(a * fps)), f1 = Math.min(numFrames, Math.max(f0 + 1, Math.ceil(b * fps)));
    const acc = new Float64Array(S);
    for (let f = f0; f < f1; f++) for (let s = 0; s < S; s++) acc[s] += probs[f * S + s];
    let best = 0; for (let s = 1; s < S; s++) if (acc[s] > acc[best]) best = s;
    return { best, mass: acc[best] / (f1 - f0) };
  };
  let prev = null;
  const out = words.map(w => {
    let r = score(w.start, w.end);
    for (let pad = 0.25; r.mass < 0.2 && pad <= 1; pad *= 2) r = score(w.start - pad, w.end + pad);
    const speaker = r.mass < 0.05 && prev != null ? prev : r.best;
    prev = speaker;
    return { ...w, speaker };
  });
  // Whisper's word times drift a few hundred ms; if a speaker change lands within 3 words
  // of a sentence end, move it to the punctuation.
  const endsSentence = i => /[.?!]["')\]]?$/.test(out[i].text.trim());
  for (let i = 1; i < out.length; i++) {
    if (out[i].speaker === out[i - 1].speaker || endsSentence(i - 1)) continue;
    const from = out[i - 1].speaker, to = out[i].speaker;
    let target = null;
    for (let d = 1; d <= 3 && target == null; d++) {
      if (i - 1 - d >= 0 && endsSentence(i - 1 - d) && out.slice(i - d, i).every(w => w.speaker === from)) target = i - d;
      else if (i - 1 + d < out.length && endsSentence(i - 1 + d) && out.slice(i, i + d).every(w => w.speaker === to)) target = i + d;
    }
    if (target == null) continue;
    if (target < i) for (let k = target; k < i; k++) out[k].speaker = to;
    else for (let k = i; k < target; k++) out[k].speaker = from;
  }
  return out;
}

// Group speaker-labelled words into turns.
export function toTurns(words) {
  const turns = [];
  for (const w of words) {
    const last = turns[turns.length - 1];
    if (last && last.speaker === w.speaker) { last.text += w.text; last.end = w.end; }
    else turns.push({ speaker: w.speaker, start: w.start, end: w.end, text: w.text });
  }
  turns.forEach(t => t.text = t.text.trim());
  return turns;
}
