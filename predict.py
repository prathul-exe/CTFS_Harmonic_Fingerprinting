"""
predict.py
----------
Give it any audio file → it tells you the instrument.

Usage:
    py predict.py <audio_file>
"""

import sys
import json
import numpy as np
from pathlib import Path
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
import csv

BASE_DIR      = Path("E:/CTFS_Harmonic_Fingerprinting")
FEAT_DIR      = BASE_DIR / "data" / "features"
CTFS_DIR      = BASE_DIR / "data" / "ctfs"
N_HARMONICS   = 32
FRAME_MS      = 100
TARGET_SR     = 22050
ENERGY_THRESH = 0.01
INSTRUMENTS   = ["sitar", "veena", "guitar"]
MAX_DURATION  = 30.0


def load_audio(path):
    import librosa
    print(f"  Loading first {MAX_DURATION:.0f} seconds...")
    x, sr = librosa.load(str(path), sr=TARGET_SR, mono=True,
                         duration=MAX_DURATION)
    x = x / (np.max(np.abs(x)) + 1e-9)
    return x, TARGET_SR


def estimate_f0(frame, sr):
    frame = frame - np.mean(frame)
    if np.max(np.abs(frame)) < 1e-6:
        return None
    corr    = np.correlate(frame, frame, mode='full')
    corr    = corr[len(corr)//2:]
    min_lag = int(sr / 2000)
    max_lag = int(sr / 50)
    corr_s  = corr[min_lag:max_lag]
    if len(corr_s) == 0:
        return None
    peak_lag = np.argmax(corr_s) + min_lag
    return sr / peak_lag


def compute_ctfs(frame, sr, f0):
    X     = np.fft.fft(frame)
    freqs = np.fft.fftfreq(len(frame), d=1/sr)
    mags  = np.zeros(N_HARMONICS)
    for n in range(1, N_HARMONICS+1):
        idx       = np.argmin(np.abs(freqs - n*f0))
        mags[n-1] = np.abs(X[idx]) / len(frame)
    if mags[0] > 1e-9:
        mags = mags / mags[0]
    return mags


def extract_features(x, sr):
    frame_len  = int(FRAME_MS * 1e-3 * sr)
    hop_len    = int(frame_len * 0.5)
    hann_win   = np.hanning(frame_len)
    num_frames = (len(x) - frame_len) // hop_len + 1
    all_mags   = []
    for i in range(num_frames):
        start = i * hop_len
        frame = x[start: start + frame_len]
        if len(frame) < frame_len:
            continue
        fw  = frame * hann_win
        rms = np.sqrt(np.mean(fw**2))
        if rms < ENERGY_THRESH:
            continue
        f0 = estimate_f0(fw, sr)
        if f0 is None or f0 < 50 or f0 > 2000:
            continue
        mags = compute_ctfs(fw, sr, f0)
        all_mags.append(mags)
    return np.array(all_mags) if all_mags else None


def load_training_data():
    feat_path = FEAT_DIR / "feature_matrix.csv"
    X, y = [], []
    with open(feat_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            X.append([float(row[f"h{n}"]) for n in range(1, 33)])
            y.append(row["instrument"])
    return np.array(X), np.array(y)


def predict(audio_path):
    print("\n" + "="*50)
    print("  CTFS HARMONIC FINGERPRINT CLASSIFIER")
    print("="*50)
    print(f"\n  Input: {audio_path}")

    x, sr = load_audio(audio_path)
    print(f"  Loaded {len(x)/sr:.1f} seconds")

    print("  Extracting CTFS features...")
    mags = extract_features(x, sr)
    if mags is None or len(mags) == 0:
        print("  ERROR: No valid frames found.")
        sys.exit(1)
    print(f"  {len(mags)} valid frames extracted")

    # ── KEY FIX: use raw magnitudes as features ───────────────────
    # Raw magnitudes work better for prediction on unseen audio
    # because deviation vectors depend on knowing the instrument class
    X_test           = mags
    X_train, y_train = load_training_data()

    # Reconstruct raw magnitudes from training data too
    # (undo the deviation: raw = dev + mean_env)
    with open(FEAT_DIR / "mean_envelopes.json") as f:
        env = json.load(f)
    mean_envs = {inst: np.array(env[inst]) for inst in INSTRUMENTS}

    # Add back mean envelope per class to get raw magnitudes
    X_train_raw = []
    for feat, label in zip(X_train, y_train):
        X_train_raw.append(feat + mean_envs[label])
    X_train_raw = np.array(X_train_raw)

    # Scale
    scaler      = StandardScaler()
    X_tr_sc     = scaler.fit_transform(X_train_raw)
    X_te_sc     = scaler.transform(X_test)

    # Train classifiers
    svm = SVC(kernel='rbf', C=10, gamma='scale',
              probability=True, random_state=42)
    knn = KNeighborsClassifier(n_neighbors=5)
    svm.fit(X_tr_sc, y_train)
    knn.fit(X_tr_sc, y_train)

    # Predict
    svm_preds = svm.predict(X_te_sc)
    knn_preds = knn.predict(X_te_sc)

    def vote(preds):
        counts = {i: int(np.sum(preds == i)) for i in INSTRUMENTS}
        return max(counts, key=counts.get), counts

    svm_result, svm_counts = vote(svm_preds)
    knn_result, knn_counts = vote(knn_preds)

    svm_proba  = svm.predict_proba(X_te_sc)
    mean_proba = np.mean(svm_proba, axis=0)
    classes    = list(svm.classes_)

    # Final = kNN (higher accuracy)
    final = knn_result

    print("\n" + "="*50)
    print("  PREDICTION RESULTS")
    print("="*50)
    print(f"\n  kNN  (96.45% acc) → {knn_result.upper()}")
    print(f"  SVM  (95.43% acc) → {svm_result.upper()}")
    print(f"\n  ✅ FINAL PREDICTION : {final.upper()}")

    print("\n  Confidence (SVM probabilities):")
    proba_dict = dict(zip(classes, mean_proba))
    for inst in sorted(INSTRUMENTS, key=lambda i: -proba_dict.get(i,0)):
        prob = proba_dict.get(inst, 0)
        bar  = "█" * int(prob * 30)
        print(f"    {inst:<8} {prob*100:5.1f}%  {bar}")

    print("\n  kNN Frame votes:")
    total = sum(knn_counts.values())
    for inst in INSTRUMENTS:
        count = knn_counts.get(inst, 0)
        pct   = 100 * count / total if total > 0 else 0
        bar   = "█" * int(pct / 3)
        print(f"    {inst:<8} {count:>5} frames  ({pct:.1f}%)  {bar}")

    print("\n" + "="*50)
    return final


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: py predict.py <audio_file>")
        sys.exit(1)
    audio_path = Path(sys.argv[1])
    if not audio_path.exists():
        print(f"File not found: {audio_path}")
        sys.exit(1)
    predict(audio_path)