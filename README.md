# Local Interview Transcriber

Transcribe research interviews and label who is speaking, **entirely in your browser**. There's no server, no upload and no sign-in, and the audio never leaves your computer.

**Use it:** https://nealcaren.github.io/local-interview-transcriber/ (works best in Chrome or Edge on a laptop or desktop)

Built for graduate students and researchers working with sensitive recordings (oral histories, qualitative interviews, focus groups). Instead of a Python setup or a cloud service, you open a web page.

## What it does

1. **Add a recording.** Drag in an audio or video file, or record in the browser.
2. **Choose what you need.** Transcribe the words, identify the speakers, or both.
3. **Check and correct.** Click a timestamp to hear that passage, fix text in place, rename speakers ("Interviewer", "Participant 07"), or move a turn to a different speaker. <kbd>Esc</kbd> plays and pauses.
4. **Export:**

| For | Format |
|---|---|
| Reading, coding, NVivo / ATLAS.ti / MAXQDA | Word (`.docx`), plain text (`.txt`) |
| Playing alongside audio or video | Subtitles (`.srt`), WebVTT captions with speaker tags (`.vtt`) |
| Analysis | Spreadsheet (`.csv`, one row per turn), JSON (turns plus every word with its timing and speaker), RTTM (standard diarization format) |

The first run downloads the models: about 670 MB with a GPU (Whisper large-v3-turbo) or 350 MB on a CPU (Whisper small). Your browser keeps them, so later runs start right away.

## How it works

| Step | Model | Runs on |
|---|---|---|
| Words | OpenAI Whisper, large-v3-turbo ("Most accurate", GPU only) or small ("Fast"), from the prebuilt `onnx-community/*_timestamped` exports, with word timestamps | [transformers.js](https://github.com/huggingface/transformers.js), WebGPU or WASM |
| Speakers | [NVIDIA Nemotron 3 Diarization](https://huggingface.co/nvidia/Nemotron-3-Diarization) (Sortformer, up to 8 speakers, 10 ms resolution), converted to ONNX: [NealCaren/Nemotron-3-Diarization-ONNX](https://huggingface.co/NealCaren/Nemotron-3-Diarization-ONNX) | [ONNX Runtime Web](https://onnxruntime.ai), int8 on WASM |
| Alignment | Each Whisper word goes to the speaker with the most probability mass over the word's span. A speaker change within 3 words of a sentence end is moved to the punctuation. | plain JS |

Whisper and the diarizer run in parallel in separate Web Workers. `diar.js` is a dependency-free port of the model's log-mel front end, chunking and arrival-order speaker cache from the `transformers` implementation. With the fp32 model it matches the reference output exactly (100% of frame decisions on the test clip); the int8 model shipped by default agrees on 99.994% of frames.

How well it works on a real interview: part one of Southern Oral History Program interview E-0055 (65 minutes of 1974 cassette audio, two speakers) took 9 minutes on an Apple M3 Max with the defaults, including first-time model downloads. Compared with the archive's typed transcript, 98.8% of matched words were attributed to the right speaker (99.5% for the narrator, 94.4% for the interviewer, whose short questions are harder). Whisper produced 88% of the typist's words exactly; the typed version is lightly edited, so the true error rate is lower.

## Files

| File | What it is |
|---|---|
| `index.html`, `app.js` | The page and its UI logic |
| `diar.js` | Nemotron diarization pipeline: features, chunked inference, speaker cache, word alignment |
| `export.js` | Export formats, including a small `.docx` writer |
| `worker.js`, `whisper-worker.js` | Diarizer and Whisper workers |
| `coi-serviceworker.js` | Adds cross-origin isolation on GitHub Pages so WASM can use multiple threads ([coi-serviceworker](https://github.com/gzuidhof/coi-serviceworker), MIT) |
| `fonts/` | Courier Prime (OFL) and Special Elite (Apache 2.0), self-hosted so the page makes no requests to Google |
| `sample.m4a` | First five minutes of Southern Oral History Program interview [E-0055](https://dc.lib.unc.edu/cdm/compoundobject/collection/sohp/id/4477) (Arthur J. Beaumont, interviewed by Derek Williams, 1974), Wilson Library, UNC-Chapel Hill. Rebuilt by `tools/make_sample.py` |
| `tools/` | ONNX export and quantization, reference/parity test, local server |

## Development

```sh
python3 tools/serve.py        # http://127.0.0.1:8765
```

By default the page loads the diarizer from Hugging Face. To test locally built models, run `tools/export_onnx.py` and `tools/quantize.py` (which write `models/`, ignored by git) and open `http://127.0.0.1:8765/?models=local`.

To check `diar.js` against the `transformers` reference:

```sh
python tools/make_reference.py clip.wav   # any 16 kHz mono wav
npm i onnxruntime-node && node tools/parity_test.mjs step step_int8
```

## Limitations

- Automatic transcripts contain errors, especially names, numbers and overlapping speech. Check them against the audio before quoting.
- Similar voices can be merged into one speaker, and the model handles at most 8 speakers. Synthetic text-to-speech voices are especially hard for it to tell apart.
- Long recordings use a lot of memory. The speaker step works in small pieces, but Whisper's memory grows as it goes (in Chrome, about 0.6 GB for 5 minutes and 1.2 GB for 65 minutes, a limitation of transformers.js 4.3.0), on top of the decoded audio (about 0.25 GB per hour). For multi-hour files, or if Safari reloads the page, tick "One step at a time" under Advanced settings or split the recording into one-hour parts. Chrome and Edge tolerate more memory than Safari; the page recommends them to Safari users.

## Privacy

All processing happens on your device. The only network requests are downloads of the page and the models. Check that local processing fits your IRB protocol and consent forms. This repo contains only the app, and `.gitignore` blocks audio and transcript files.

## Licenses

The Nemotron 3 Diarization weights are © NVIDIA and licensed under [OpenMDW-1.1](https://huggingface.co/NealCaren/Nemotron-3-Diarization-ONNX/blob/main/LICENSE). Whisper is MIT-licensed (OpenAI).
