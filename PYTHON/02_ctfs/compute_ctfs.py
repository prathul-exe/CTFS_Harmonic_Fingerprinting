"""
compute_ctfs.py
---------------
Module 2 — CTFS Harmonic Analysis

For each instrument:
1. Load WAV file
2. Read pakad region from JSON annotation
3. Cut into 100ms Hann-windowed frames
4. Estimate f0 per frame via autocorrelation
5. Compute N=32 CTFS harmonic coefficients via DFT
6. Extract normalized harmonic envelope |cn|
7. Save results to data/ctfs/

Usage:
    python python/02_ctfs/compute_ctfs.py
"""

import os
import json
import numpy as np
import librosa
from pathlib import Path

# ── Parameters ────────────────────────────────────────────────────────
N_HARMONICS  = 32        # number of CTFS coefficients to extract
FRAME_MS     = 100       # frame duration in milliseconds
OVERLAP_PCT  = 0.50      # 50% overlap
TARGET_SR    = 22050     # sample rate
ENERGY_THRESH = 0.01     # silence removal threshold

BASE_DIR     = Path("E:/CTFS_Harmonic_Fingerprinting")
RAW_DIR      = BASE_DIR / "data" / "raw"
ANNOT_DIR    = BASE_DIR / "data" / "annotations"
OUT_DIR      = BASE_DIR / "data" / "ctfs"

INSTRUMENTS  = {
    "sitar" : ["yaman", "bhairavi"],
    "veena" : ["bhairavi", "desh"],
    "guitar": []   # no raga annotations for guitar
}


# ── Helper: Estimate f0 via autocorrelation ───────────────────────────
def estimate_f0(frame, sr, f0_min=80, f0_max=1200):
    """
    Estimate fundamental frequency of a frame using autocorrelation.
    Returns f0 in Hz.
    """
    # Autocorrelation
    corr = np.correlate(frame, frame, mode='full')
    corr = corr[len(corr)//2:]   # take positive lags only

    # Search range in samples
    lag_min = int(sr / f0_max)
    lag_max = int(sr / f0_min)
    lag_max = min(lag_max, len(corr) - 1)

    # Find peak in search range
    search = corr[lag_min:lag_max]
    if len(search) == 0:
        return None
    peak_lag = np.argmax(search) + lag_min
    f0 = sr / peak_lag
    return f0


# ── Helper: Compute CTFS coefficients ────────────────────────────────
def compute_ctfs_coefficients(frame, sr, f0, N=32):
    """
    Compute N complex CTFS coefficients using DFT.

    The nth coefficient corresponds to harmonic at frequency n*f0.
    Returns:
        magnitudes : np.ndarray shape (N,) — |cn| normalized by |c1|
        phases     : np.ndarray shape (N,) — angle(cn) in radians
    """
    n_samples = len(frame)
    T0        = 1.0 / f0          # fundamental period in seconds
    t         = np.arange(n_samples) / sr   # time axis

    magnitudes = np.zeros(N)
    phases     = np.zeros(N)

    for n in range(1, N + 1):
        # CTFS coefficient: cn = (1/T0) * integral(x(t) * e^(-jnw0t) dt)
        omega_n   = 2 * np.pi * n * f0
        kernel    = np.exp(-1j * omega_n * t)
        cn        = np.trapz(frame * kernel, t) / T0
        magnitudes[n-1] = np.abs(cn)
        phases[n-1]     = np.angle(cn)

    # Normalize by fundamental (c1) magnitude
    if magnitudes[0] > 1e-9:
        magnitudes = magnitudes / magnitudes[0]

    return magnitudes, phases


# ── Main Processing Function ──────────────────────────────────────────
def process_instrument(instrument, raga=None):
    """
    Process one instrument+raga combination.
    Returns list of feature dicts.
    """
    # Load audio
    if raga:
        audio_path = RAW_DIR / instrument / f"{raga}.wav"
        annot_path = ANNOT_DIR / f"{instrument}_{raga}.json"
    else:
        # Guitar — process all wav files
        audio_path = None
        annot_path = None

    results = []

    # ── Guitar: process all files without annotation ──────────────
    if instrument == "guitar":
        guitar_dir = RAW_DIR / "guitar"
        wav_files  = sorted(guitar_dir.glob("*.wav"))

        for wav_path in wav_files:
            x, sr = librosa.load(str(wav_path), sr=TARGET_SR, mono=True)
            x = x / (np.max(np.abs(x)) + 1e-9)
            frames_result = extract_frames_and_ctfs(x, sr, instrument, "none", str(wav_path.name))
            results.extend(frames_result)
        return results

    # ── Sitar/Veena: use pakad annotation ─────────────────────────
    if not audio_path.exists():
        print(f"  ❌ Audio not found: {audio_path}")
        return []

    if not annot_path.exists():
        print(f"  ❌ Annotation not found: {annot_path}")
        return []

    # Load annotation
    with open(annot_path) as f:
        annot = json.load(f)

    pakad_start = annot["pakad"]["start"]
    pakad_end   = annot["pakad"]["end"]

    print(f"  Pakad region: {pakad_start}s → {pakad_end}s")

    # Load audio
    x, sr = librosa.load(str(audio_path), sr=TARGET_SR, mono=True,
                          offset=pakad_start, duration=pakad_end - pakad_start)
    x = x / (np.max(np.abs(x)) + 1e-9)

    results = extract_frames_and_ctfs(x, sr, instrument, raga, f"{raga}.wav")
    return results


# ── Frame Extraction + CTFS ───────────────────────────────────────────
def extract_frames_and_ctfs(x, sr, instrument, raga, filename):
    """
    Cut signal into frames, estimate f0, compute CTFS per frame.
    """
    frame_len  = int(FRAME_MS * 1e-3 * sr)
    hop_len    = int(frame_len * (1 - OVERLAP_PCT))
    hann_win   = np.hanning(frame_len)
    num_frames = (len(x) - frame_len) // hop_len + 1

    results = []

    for i in range(num_frames):
        start = i * hop_len
        frame = x[start: start + frame_len]

        # Skip silent frames
        rms = np.sqrt(np.mean(frame ** 2))
        if rms < ENERGY_THRESH:
            continue

        # Apply Hann window
        frame_win = frame * hann_win

        # Estimate f0
        f0 = estimate_f0(frame_win, sr)
        if f0 is None or f0 < 80 or f0 > 1200:
            continue

        # Compute CTFS
        magnitudes, phases = compute_ctfs_coefficients(frame_win, sr, f0, N=N_HARMONICS)

        results.append({
            "instrument"  : instrument,
            "raga"        : raga,
            "filename"    : filename,
            "frame_idx"   : i,
            "f0_hz"       : f0,
            "magnitudes"  : magnitudes.tolist(),
            "phases"      : phases.tolist()
        })

    print(f"    {filename}: {len(results)} valid frames extracted")
    return results


# ── Runner ────────────────────────────────────────────────────────────
def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    all_results = {}

    for instrument, ragas in INSTRUMENTS.items():
        print(f"\n{'='*45}")
        print(f"Processing: {instrument.upper()}")
        print(f"{'='*45}")

        inst_results = []

        if instrument == "guitar":
            inst_results = process_instrument("guitar")
        else:
            for raga in ragas:
                print(f"\n  Raga: {raga}")
                r = process_instrument(instrument, raga)
                inst_results.extend(r)

        all_results[instrument] = inst_results

        # Save per instrument
        out_path = OUT_DIR / f"{instrument}_ctfs.json"
        with open(out_path, "w") as f:
            json.dump(inst_results, f)
        print(f"\n  ✅ Saved {len(inst_results)} frames → {out_path}")

    # Summary
    print(f"\n{'='*45}")
    print("CTFS EXTRACTION SUMMARY")
    print(f"{'='*45}")
    for inst, res in all_results.items():
        print(f"  {inst:<8}: {len(res)} frames")
    print(f"{'='*45}")
    print("Done! Proceed to Module 3 (Fidelity Analysis)")


if __name__ == "__main__":
    main()