// Speaker diarization worker: Nemotron 3 Diarization (ONNX) on onnxruntime-web.
// CPU-only build: its WebAssembly is about half the size of the all-backends one (14 vs 28 MB).
const ORT_VERSION = '1.30.0';
const ORT_CDN = `https://cdn.jsdelivr.net/npm/onnxruntime-web@${ORT_VERSION}/dist/`;
const ort = await import(ORT_CDN + 'ort.wasm.min.mjs');
import { Diarizer } from './diar.js';

ort.env.wasm.wasmPaths = ORT_CDN;
// a quarter of the cores: speakers still finish well ahead of Whisper, and the computer stays usable
ort.env.wasm.numThreads = self.crossOriginIsolated ? Math.max(1, Math.floor((navigator.hardwareConcurrency || 4) / 4)) : 1;

const CACHE = 'nemotron-diar-v1';
let diarizer = null, loadedKey = null;

// fetch with progress, kept in Cache Storage so later visits skip the download
async function fetchBytes(url, label, report) {
  let cache = null;
  try { cache = await caches.open(CACHE); } catch { /* storage unavailable */ }
  const hit = cache && await cache.match(url);
  if (hit) return new Uint8Array(await hit.arrayBuffer());
  const net = await fetch(url);
  if (!net.ok) throw new Error(`Could not download ${label} (HTTP ${net.status})`);
  const total = +net.headers.get('content-length') || 0;
  const reader = net.body.getReader();
  // write straight into one buffer when the size is known, to avoid holding extra copies
  let buf = new Uint8Array(total || 1 << 20), got = 0;
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    if (got + value.length > buf.length) { const bigger = new Uint8Array(Math.max(buf.length * 2, got + value.length)); bigger.set(buf.subarray(0, got)); buf = bigger; }
    buf.set(value, got); got += value.length;
    report?.(got, total);
  }
  buf = got === buf.length ? buf : buf.slice(0, got);
  try { await cache?.put(url, new Response(buf)); } catch { /* over quota: skip caching */ }
  return buf;
}

async function load(base, model, backend) {
  const key = base + model + '/' + backend;
  if (key === loadedKey) return;
  const t0 = performance.now();
  const report = (got, total) => postMessage({ type: 'status', text: `Downloading the speaker model (first time only, ${(got / 1e6).toFixed(0)}/${(total / 1e6).toFixed(0)} MB)`, progress: total ? got / total : null });
  const [embedBuf, stepBuf, melBuf, silBuf] = await Promise.all([
    fetchBytes(base + 'embed.onnx', 'embedder'),
    fetchBytes(base + model + '.onnx', 'speaker model', report),
    fetchBytes(base + 'mel_filters.bin', 'mel filters'),
    fetchBytes(base + 'silence_embeds.bin', 'silence embedding'),
  ]);
  postMessage({ type: 'status', text: 'Starting the speaker model…', progress: null });
  const opts = { executionProviders: ['wasm'], graphOptimizationLevel: 'all' };
  const embed = await ort.InferenceSession.create(embedBuf, { executionProviders: ['wasm'] });
  const step = await ort.InferenceSession.create(stepBuf, opts);
  diarizer = new Diarizer(ort, embed, step, new Float32Array(melBuf.buffer), new Float32Array(silBuf.buffer));
  loadedKey = key;
  postMessage({ type: 'loaded', ms: performance.now() - t0 });
}

let lastRead = 0;
const pendingReads = new Map();

onmessage = async ({ data }) => {
  if (data.type === 'audio') { pendingReads.get(data.id)?.(data.audio); pendingReads.delete(data.id); return; }
  try {
    if (data.type === 'run') {
      await load(data.base, data.model, data.backend);
      postMessage({ type: 'status', text: 'Diarizing…', progress: 0 });
      const t0 = performance.now();
      // the page keeps the recording; ask it for each chunk's samples as they are needed
      const source = { length: data.length, read: (s0, s1) => new Promise(resolve => {
        const id = ++lastRead; pendingReads.set(id, resolve); postMessage({ type: 'audio', id, s0, s1 });
      }) };
      const { probs, numFrames } = await diarizer.run(source, p =>
        postMessage({ type: 'status', text: 'Diarizing…', progress: p }));
      postMessage({ type: 'result', probs, numFrames, ms: performance.now() - t0 }, [probs.buffer]);
    }
  } catch (e) {
    postMessage({ type: 'error', text: String(e && e.message || e) });
  }
};
