import librosa
import numpy as np
import json
import os
import sounddevice as sd

def annotate_file(audio_path, instrument, raga):
    print(f"\n{'='*50}")
    print(f"Instrument: {instrument} | Raga: {raga}")
    print(f"File: {audio_path}")
    print(f"{'='*50}")

    # Load audio
    x, sr = librosa.load(audio_path, sr=22050, mono=True)
    duration = len(x) / sr
    print(f"Duration: {duration:.1f} seconds ({duration/60:.1f} minutes)")

    # Play first 30 seconds so you can hear it
    print("\nPlaying first 30 seconds so you can hear the recording...")
    preview = x[:sr*30]
    sd.play(preview, sr)
    sd.wait()

    print("\nNow enter the pakad (main raga phrase) timestamps.")
    print("Listen to the full recording and identify where the")
    print("characteristic phrase of the raga starts and ends.")
    print("Tip: Pakad usually appears in the first 2-3 minutes (alap section)\n")

    start = float(input("Enter START time of pakad in seconds (e.g. 45.0): "))
    end   = float(input("Enter END time of pakad in seconds   (e.g. 90.0): "))

    annotation = {
        "instrument" : instrument,
        "raga"       : raga,
        "audio_file" : os.path.basename(audio_path),
        "pakad"      : {"start": start, "end": end},
        "duration"   : duration
    }

    # Save annotation
    out_dir  = "data/annotations"
    os.makedirs(out_dir, exist_ok=True)
    out_path = f"{out_dir}/{instrument}_{raga}.json"
    with open(out_path, "w") as f:
        json.dump(annotation, f, indent=4)

    print(f"\n✅ Annotation saved to: {out_path}")
    return annotation


# ── Run for all files ─────────────────────────────────────────────
if __name__ == "__main__":
    files = [
        ("E:/CTFS_Harmonic_Fingerprinting/data/raw/sitar/yaman.wav",    "sitar", "yaman"),
        ("E:/CTFS_Harmonic_Fingerprinting/data/raw/sitar/bhairavi.wav",  "sitar", "bhairavi"),
        ("E:/CTFS_Harmonic_Fingerprinting/data/raw/veena/bhairavi.wav",  "veena", "bhairavi"),
        ("E:/CTFS_Harmonic_Fingerprinting/data/raw/veena/desh.wav",      "veena", "desh"),
    ]

    for path, instrument, raga in files:
        annotate_file(path, instrument, raga)

    print("\n✅ All annotations complete! Proceed to segmentation.")
    