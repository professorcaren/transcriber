// Speaker diarization worker: Nemotron 3 Diarization (ONNX) on onnxruntime-web.
const ORT_VERSION = '1.30.0';
const ORT_CDN = `https://cdn.jsdelivr.net/npm/onnxruntime-web@${ORT_VERSION}/dist/`;
const ort = await import(ORT_CDN + 'ort.all.min.mjs');
import { Diarizer } from './diar.js';

ort.env.wasm.wasmPaths = ORT_CDN;
ort.env.wasm.numThreads = self.crossOriginIsolated ? Math.min(8, navigator.hardwareConcurrency || 4) : 1;

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
  const parts = []; let got = 0;
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    parts.push(value); got += value.length;
    report?.(got, total);
  }
  const blob = new Blob(parts);
  try { await cache?.put(url, new Response(blob)); } catch { /* over quota: skip caching */ }
  return new Uint8Array(await blob.arrayBuffer());
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
  const opts = { executionProviders: backend === 'webgpu' ? ['webgpu', 'wasm'] : ['wasm'], graphOptimizationLevel: 'all' };
  const embed = await ort.InferenceSession.create(embedBuf, { executionProviders: ['wasm'] });
  const step = await ort.InferenceSession.create(stepBuf, opts);
  diarizer = new Diarizer(ort, embed, step, new Float32Array(melBuf.buffer), new Float32Array(silBuf.buffer));
  loadedKey = key;
  postMessage({ type: 'loaded', ms: performance.now() - t0 });
}

onmessage = async ({ data }) => {
  try {
    if (data.type === 'run') {
      await load(data.base, data.model, data.backend);
      postMessage({ type: 'status', text: 'Diarizing…', progress: 0 });
      const t0 = performance.now();
      const { probs, numFrames } = await diarizer.run(data.audio, p =>
        postMessage({ type: 'status', text: 'Diarizing…', progress: p }));
      postMessage({ type: 'result', probs, numFrames, ms: performance.now() - t0 }, [probs.buffer]);
    }
  } catch (e) {
    postMessage({ type: 'error', text: String(e && e.message || e) });
  }
};
