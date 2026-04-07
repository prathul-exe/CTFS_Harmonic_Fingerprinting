"""
build_features.py
-----------------
Module 4 — Feature Vector Construction
"""

import json
import numpy as np
import csv
from pathlib import Path

N_HARMONICS  = 32
BASE_DIR     = Path("E:/CTFS_Harmonic_Fingerprinting")
CTFS_DIR     = BASE_DIR / "data" / "ctfs"
FEAT_DIR     = BASE_DIR / "data" / "features"
INSTRUMENTS  = ["sitar", "veena", "guitar"]


def load_ctfs(instrument):
    path = CTFS_DIR / f"{instrument}_ctfs.json"
    with open(path) as f:
        data = json.load(f)
    magnitudes  = np.array([d["magnitudes"] for d in data])
    raga_labels = [d["raga"] for d in data]
    return magnitudes, raga_labels


def compute_mean_envelope(magnitudes):
    return np.mean(magnitudes, axis=0)


def compute_deviation_vectors(magnitudes, mean_envelope):
    return magnitudes - mean_envelope


def build_feature_matrix():
    print("MODULE 4 — FEATURE VECTOR CONSTRUCTION")
    print("="*45)
    FEAT_DIR.mkdir(parents=True, exist_ok=True)

    all_features, all_labels, all_raga_labels = [], [], []
    mean_envelopes = {}
    raw_data = {}

    print("\nComputing mean harmonic envelopes...")
    for inst in INSTRUMENTS:
        mags, ragas          = load_ctfs(inst)
        mean_env             = compute_mean_envelope(mags)
        mean_envelopes[inst] = mean_env
        raw_data[inst]       = (mags, ragas)
        print(f"  {inst:<8}: {len(mags)} frames")

    env_path = FEAT_DIR / "mean_envelopes.json"
    with open(env_path, "w") as f:
        json.dump({inst: mean_envelopes[inst].tolist()
                   for inst in INSTRUMENTS}, f, indent=2)
    print(f"\n  Mean envelopes saved: {env_path}")

    print("\nBuilding deviation vectors...")
    for inst in INSTRUMENTS:
        mags, ragas = raw_data[inst]
        dev_vectors = compute_deviation_vectors(mags, mean_envelopes[inst])
        for dev, raga in zip(dev_vectors, ragas):
            all_features.append(dev.tolist())
            all_labels.append(inst)
            all_raga_labels.append(raga)
        print(f"  {inst:<8}: {len(dev_vectors)} vectors")

    feat_path = FEAT_DIR / "feature_matrix.csv"
    header = [f"h{n}" for n in range(1, N_HARMONICS+1)] + ["instrument", "raga"]
    with open(feat_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for feat, inst, raga in zip(all_features, all_labels, all_raga_labels):
            writer.writerow(feat + [inst, raga])

    print(f"\n  Feature matrix saved: {feat_path}")
    print(f"  Total samples : {len(all_features)}")
    print(f"  Feature dims  : {N_HARMONICS}")

    print("\n  Class Distribution:")
    print("  ╔═══════════════════════════╗")
    for inst in INSTRUMENTS:
        count = all_labels.count(inst)
        pct   = 100 * count / len(all_labels)
        print(f"  ║  {inst:<8} : {count:>4} ({pct:.1f}%)   ║")
    print("  ╚═══════════════════════════╝")


if __name__ == "__main__":
    build_feature_matrix()
    print("\n" + "="*45)
    print("MODULE 4 COMPLETE")
    print("Proceed to Module 5 — Classification")
    print("="*45)