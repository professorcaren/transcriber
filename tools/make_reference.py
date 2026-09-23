"""Reference output from the transformers implementation, for tools/parity_test.mjs.

    python tools/make_reference.py sample.wav   # writes tools/ref/*.f32
"""
import os, sys
import numpy as np, soundfile as sf, torch
from transformers import AutoModelForAudioFrameClassification, AutoProcessor

mid = "nvidia/Nemotron-3-Diarization"
proc = AutoProcessor.from_pretrained(mid)
model = AutoModelForAudioFrameClassification.from_pretrained(mid).eval()
audio, sr = sf.read(sys.argv[1], dtype="float32")
assert sr == 16000 and audio.ndim == 1, "expects 16 kHz mono"
inp = proc(audio, sampling_rate=16000)
with torch.no_grad():
    logits = model(**inp).logits[0].numpy()
os.makedirs("tools/ref", exist_ok=True)
audio.astype("<f4").tofile("tools/ref/audio.f32")
inp["input_features"][0].numpy().astype("<f4").tofile("tools/ref/feats.f32")
(1 / (1 + np.exp(-logits))).astype("<f4").tofile("tools/ref/probs.f32")
print(proc.extract_speaker_dict(torch.from_numpy(logits)[None])[0][:10])
