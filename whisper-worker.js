// Whisper transcription with word timestamps, via transformers.js (own onnxruntime, own worker).
// The page sends one 30-second window at a time, so this worker never holds the whole recording.
//   {type: 'load', model, device, dtype}          -> {type: 'loaded'}
//   {type: 'window', audio, offset, language}     -> {type: 'window', words: [{text, start, end}]}  (seconds, absolute)
import { pipeline, env } from 'https://cdn.jsdelivr.net/npm/@huggingface/transformers@4.3.0';

env.allowLocalModels = false;
let asr = null, loadedKey = null;

async function load(model, device, dtype) {
  const key = model + '|' + device + '|' + JSON.stringify(dtype);
  if (key === loadedKey) return;
  await asr?.dispose();
  asr = null; loadedKey = null;
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
    if (data.type === 'load') {
      await load(data.model, data.device, data.dtype);
      postMessage({ type: 'loaded' });
    } else if (data.type === 'window') {
      const out = await asr(data.audio, { return_timestamps: 'word', language: data.language || null, task: 'transcribe' });
      const words = (out.chunks || []).map(c => ({
        text: c.text,
        start: data.offset + (c.timestamp[0] ?? 0),
        end: data.offset + (c.timestamp[1] ?? c.timestamp[0] ?? 0),
      }));
      postMessage({ type: 'window', words });
    }
  } catch (e) {
    postMessage({ type: 'error', text: String(e && e.message || e) });
  }
};
