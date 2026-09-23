// Compare diar.js + ONNX against the transformers reference (run tools/make_reference.py first).
//   npm i onnxruntime-node && node tools/parity_test.mjs step_int8 step
import ort from 'onnxruntime-node';
import fs from 'fs';
import { Diarizer, logMel, toSegments, CFG } from '../diar.js';
const f32 = p => { const b = fs.readFileSync(p); return new Float32Array(b.buffer, b.byteOffset, b.length / 4); };
const audio = f32(new URL('ref/audio.f32', import.meta.url)), mel = f32(new URL('../models/mel_filters.bin', import.meta.url)), sil = f32(new URL('../models/silence_embeds.bin', import.meta.url));
const refF = f32(new URL('ref/feats.f32', import.meta.url)), refP = f32(new URL('ref/probs.f32', import.meta.url));
const { data, numFrames } = logMel(audio, mel);
let fd = 0; for (let i = 0; i < data.length; i++) fd = Math.max(fd, Math.abs(data[i] - refF[i]));
console.log('frames', numFrames, 'ref rows', refF.length / 128, 'feat maxdiff', fd.toFixed(5));
const embed = await ort.InferenceSession.create(new URL('../models/embed.onnx', import.meta.url).pathname);
for (const name of process.argv.slice(2)) {
  const step = await ort.InferenceSession.create(new URL(`../models/${name}.onnx`, import.meta.url).pathname);
  const d = new Diarizer(ort, embed, step, mel, sil);
  const t0 = Date.now();
  const { probs, numFrames: n } = await d.run(audio);
  const ms = Date.now() - t0;
  let md = 0, agree = 0;
  for (let i = 0; i < n * 8; i++) { md = Math.max(md, Math.abs(probs[i] - refP[i])); if ((probs[i] > .5) === (refP[i] > .5)) agree++; }
  console.log(`${name}: ${ms} ms, prob maxdiff ${md.toFixed(4)}, decision agreement ${(100 * agree / (n * 8)).toFixed(3)}%`);
  console.log(toSegments(probs, n, 0.5, 0.3).map(s => `${s.start}-${s.end}:S${s.speaker}`).join('  '));
}
