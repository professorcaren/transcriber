"""Rebuild sample.m4a: 26:58-30:53 of Southern Oral History Program interview K-0838.

Quinton E. Baker, interviewed by Chris McGinnis on February 23, 2002, in Hillsborough, North Carolina,
about his civil rights activism in the 1960s. Southern Oral History Program Collection (#4007),
Wilson Library, UNC-Chapel Hill: https://docsouth.unc.edu/sohp/K-0838/menu.html

    python tools/make_sample.py      # macOS (uses afconvert); standard library only

The archive's MP3 is 271 MB of constant 256 kbps audio, so only the bytes around the clip are fetched.
"""
import os, subprocess, tempfile, urllib.request, wave

AUDIO = "https://docsouth.unc.edu/sohp/K-0838/K-0838.mp3"
START, DURATION = 26 * 60 + 58, 3 * 60 + 55   # 00:26:58, for 3:55
TAG, RATE, PAD = 1024, 256000 // 8, 2          # ID3 tag bytes, bytes per second, seconds of slack each side

def fetch(a, b):
    req = urllib.request.Request(AUDIO, headers={"Range": f"bytes={a}-{b - 1}"})
    with urllib.request.urlopen(req) as r:
        return r.read()

with tempfile.TemporaryDirectory() as tmp:
    mp3, full, clip = (os.path.join(tmp, n) for n in ("part.mp3", "part.wav", "clip.wav"))
    # keep the tag so the decoder recognizes the stream, then the clip plus some slack
    begin = TAG + (START - PAD) * RATE
    with open(mp3, "wb") as f:
        f.write(fetch(0, TAG) + fetch(begin, begin + (DURATION + 2 * PAD) * RATE))
    subprocess.run(["afconvert", "-f", "WAVE", "-d", "LEI16@16000", "-c", "1", mp3, full], check=True)
    with wave.open(full) as w, wave.open(clip, "wb") as out:
        out.setparams(w.getparams())
        w.setpos(PAD * w.getframerate())
        out.writeframes(w.readframes(DURATION * w.getframerate()))
    subprocess.run(["afconvert", "-f", "m4af", "-d", "aac", "-b", "48000", clip, "sample.m4a"], check=True)
print("wrote sample.m4a")
