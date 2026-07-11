# Local Interview Transcriber

Transcribe **and** diarize interviews **entirely in your browser** — no server, no upload, no sign-in. The audio never leaves your machine.

Built for oral-history and research interviews where recordings are sensitive: everything runs client-side via WebGPU (with a WASM fallback), so you can hand a colleague a URL instead of a Python setup, and no cloud service ever sees the audio.

## Use it

Open the hosted page (see **Hosting** below) or run locally:

```sh
python3 -m http.server 8765   # from this folder
# then open http://localhost:8765
```

1. Pick a Whisper model (base = fast; `large-v3-turbo` = best, larger download).
2. Set **Speakers** to `2` for a standard interview (interviewer + narrator).
3. Drag in an audio/video file and click **Transcribe**.

First run downloads the models (~180 MB, cached by the browser afterward). A 30-minute interview processes in roughly 2–3 minutes on a recent Mac with WebGPU.

## How it works

| Stage | Model | Runs |
|-------|-------|------|
| Transcription | Whisper (`Xenova/whisper-*` / `onnx-community/whisper-large-v3-turbo`) | transformers.js, WebGPU/WASM |
| Speaker embeddings | WeSpeaker (`onnx-community/wespeaker-voxceleb-resnet34-LM`) | transformers.js |
| Diarization | energy-gated 2 s windows → spherical k-means → **word-level** speaker assignment | pure JS |

No pyannote, no gated model downloads, no API keys.

## Hosting (GitHub Pages)

This is a static site — GitHub Pages serves it directly:

1. Push this repo to GitHub.
2. **Settings → Pages → Build and deployment → Deploy from a branch → `main` / root.**
3. Your app is live at `https://<user>.github.io/local-interview-transcriber/`.

`.nojekyll` is included so Pages serves files as-is. HTTPS (required for WebGPU) is automatic.

**Note:** GitHub Pages can't set `Cross-Origin-Embedder-Policy`, so multi-threaded WASM isn't available there — but **WebGPU (the default, fast path) is unaffected**. For threaded WASM, host on Cloudflare Pages / Netlify with a `_headers` file setting COOP/COEP.

## Limitations

- Diarization is credible but not perfect (~2 s time resolution); skim and correct speaker labels.
- Transcription quality tracks the chosen Whisper model — use `whisper-small` or `large-v3-turbo` for archival work.
- Needs a browser with WebGPU (mainstream in 2026) or falls back to slower WASM.

## Privacy

All inference is on-device. This repo contains **only the app** — never commit interview audio or transcripts (`.gitignore` blocks them).
