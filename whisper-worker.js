// Whisper transcription with word timestamps, via transformers.js (own onnxruntime, own worker).
import { pipeline, env } from 'https://cdn.jsdelivr.net/npm/@huggingface/transformers@4.3.0';

env.allowLocalModels = false;
let asr = null, loadedKey = null;

async function load(model, device, dtype) {
  const key = model + '|' + device + '|' + JSON.stringify(dtype);
  if (key === loadedKey) return;
  const files = {};
  asr = await pipeline('automatic-speech-recognition', model, {
    device,
    dtype,
    progress_callback: p => {
      if (p.status === 'progress' && p.total) {
        files[p.file] = [p.loaded, p.total];
        const [got, tot] = Object.values(files).reduce((a, [l, t]) => [a[0] + l, a[1] + t], [0, 0]);
        postMessage({ type: 'status', text: `Downloading Whisper (${(got / 1e6).toFixed(0)}/${(tot / 1e6).toFixed(0)} MB)`, progress: got / tot });
      }
    },
  });
  loadedKey = key;
}

onmessage = async ({ data }) => {
  try {
    await load(data.model, data.device, data.dtype);
    const total = data.audio.length / 16000;
    const t0 = performance.now();
    postMessage({ type: 'status', text: 'Transcribing…', progress: 0 });
    // transcribe in 30 s windows ourselves so we can report progress and stream partial words
    const WIN = 30, STRIDE = 5;
    const words = [];
    let lastEnd = 0;
    for (let start = 0; start < total; start += WIN - 2 * STRIDE) {
      const s = Math.max(0, start - STRIDE), e = Math.min(total, start + WIN - STRIDE);
      const clip = data.audio.subarray(Math.floor(s * 16000), Math.floor(e * 16000));
      if (clip.length < 1600) break;
      const out = await asr(clip, {
        return_timestamps: 'word', language: data.language || null, task: 'transcribe',
      });
      // keep the words whose midpoint falls in this window's own (non-stride) span
      const own0 = start === 0 ? 0 : start, own1 = e >= total ? total : start + WIN - 2 * STRIDE;
      for (const c of out.chunks || []) {
        const ws = s + (c.timestamp[0] ?? 0), we = s + (c.timestamp[1] ?? c.timestamp[0] ?? 0);
        const mid = (ws + we) / 2;
        if (mid >= own0 && mid < own1 && ws >= lastEnd - 0.05) { words.push({ text: c.text, start: ws, end: we }); lastEnd = we; }
      }
      const done = Math.min(1, own1 / total);
      postMessage({ type: 'partial', words: words.slice(), progress: done,
        text: `Transcribing… ${Math.round(100 * done)}% (${(own1 / ((performance.now() - t0) / 1000)).toFixed(1)}× real time)` });
      if (e >= total) break;
    }
    postMessage({ type: 'result', words, ms: performance.now() - t0 });
  } catch (e) {
    postMessage({ type: 'error', text: String(e && e.message || e) });
  }
};
