"""Export nvidia/Nemotron-3-Diarization to ONNX (embed.onnx + step.onnx) from its transformers implementation.

    uv pip install torch onnx onnxruntime "transformers @ git+https://github.com/huggingface/transformers"
    python tools/export_onnx.py && python tools/quantize.py

Writes ./models/. The speaker cache, chunking and log-mel front end live in diar.js.
"""
import os
import numpy as np
import torch
from transformers import AutoModelForAudioFrameClassification

os.makedirs("models", exist_ok=True)
m = AutoModelForAudioFrameClassification.from_pretrained("nvidia/Nemotron-3-Diarization", attn_implementation="eager").eval()


class Embed(torch.nn.Module):
    def __init__(s):
        super().__init__(); s.e = m.model.audio_tower.embedder

    def forward(s, feats):
        return s.e(feats)


class Step(torch.nn.Module):
    """One chunk step: [speaker cache, FIFO, chunk, right context] embeddings -> speaker logits at 10 ms."""
    def __init__(s):
        super().__init__(); s.m = m.model; s.c = m.classifier

    def forward(s, embeds):
        pos = torch.arange(embeds.shape[1], device=embeds.device)[None, :]  # positions restart every step
        return s.c(s.m(inputs_embeds=embeds, position_ids=pos).last_hidden_state)


with torch.no_grad():
    torch.onnx.export(Embed(), (torch.randn(1, 340 * 8, 128),), "models/embed.onnx", input_names=["features"],
                      output_names=["embeds"], dynamic_axes={"features": {1: "n"}, "embeds": {1: "t"}}, opset_version=17, dynamo=False)
    torch.onnx.export(Step(), (torch.randn(1, 420, 512),), "models/step.onnx", input_names=["embeds"],
                      output_names=["logits"], dynamic_axes={"embeds": {1: "t"}, "logits": {1: "t8"}}, opset_version=17, dynamo=False)
m.silence_embeds.detach().numpy().astype("<f4").tofile("models/silence_embeds.bin")

import onnxruntime as ort
x = torch.randn(1, 380, 512)
with torch.no_grad():
    ref = Step()(x).numpy()
out = ort.InferenceSession("models/step.onnx").run(None, {"embeds": x.numpy()})[0]
print("step.onnx max abs diff vs torch:", np.abs(ref - out).max())
