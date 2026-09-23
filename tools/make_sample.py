"""Rebuild sample.m4a: the first five minutes of Southern Oral History Program interview E-0055.

Arthur J. Beaumont, former UNC security chief, interviewed by Derek Williams on November 17, 1974,
about the 1969 UNC food workers' strikes. Southern Oral History Program Collection (#4007),
Wilson Library, UNC-Chapel Hill: https://dc.lib.unc.edu/cdm/compoundobject/collection/sohp/id/4477
("No restrictions. Open to research.")

    python tools/make_sample.py      # macOS (uses afconvert for AAC); needs librosa and soundfile
"""
import os, subprocess, tempfile, urllib.request
import librosa, soundfile as sf

AUDIO_01 = "https://dc.lib.unc.edu/utils/getstream/collection/sohp/id/4473"

with tempfile.TemporaryDirectory() as tmp:
    mp3, wav = os.path.join(tmp, "audio01.mp3"), os.path.join(tmp, "clip.wav")
    urllib.request.urlretrieve(AUDIO_01, mp3)
    audio, sr = librosa.load(mp3, sr=16000, mono=True, duration=300)
    sf.write(wav, audio, sr, subtype="PCM_16")
    subprocess.run(["afconvert", "-f", "m4af", "-d", "aac", "-b", "48000", wav, "sample.m4a"], check=True)
print("wrote sample.m4a")
