"""
predict.py
----------
Give it any audio file → it tells you the instrument.

Usage:
    py predict.py <path_to_audio_file>

Example:
    py predict.py test_audio/unknown_sitar.wav
    py predict.py test_audio/mystery.mp3
"""

import sys
import json
import numpy as np
from pathlib import Path
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
import csv

# ── Parameters ────────────────────────────────────────────────────
BASE_DIR     = Path("E:/CTFS_Harmonic_Fingerprinting")
FEAT_DIR     = BASE_DIR / "data" / "features"
CTFS_DIR     = BASE_DIR / "data" / "ctfs"
N_HARMONICS  = 32
FRAME_MS     = 100
TARGET_SR    = 22050
ENERGY_THRESH = 0.01
INSTRUMENTS  = ["sitar", "veena", "guitar"]


# ── Step 1: Load and convert audio ────────────────────────────────
def load_audio(path):
    try:
        import librosa
        x, sr = librosa.load(str(path), sr=TARGET_SR, mono=True)
        x = x / (np.max(np.abs(x)) + 1e-9)
        return x, TARGET_SR
    except Exception as e:
        print(f"  Error loading audio: {e}")
        sys.exit(1)


# ── Step 2: Extract CTFS features ─────────────────────────────────
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
    X      = np.fft.fft(frame)
    freqs  = np.fft.fftfreq(len(frame), d=1/sr)
    mags   = np.zeros(N_HARMONICS)
    for n in range(1, N_HARMONICS+1):
        idx      = np.argmin(np.abs(freqs - n*f0))
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


# ── Step 3: Load training data + retrain ──────────────────────────
def load_training_data():
    feat_path = FEAT_DIR / "feature_matrix.csv"
    X, y = [], []
    with open(feat_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            X.append([float(row[f"h{n}"]) for n in range(1, 33)])
            y.append(row["instrument"])
    return np.array(X), np.array(y)


def load_mean_envelopes():
    with open(FEAT_DIR / "mean_envelopes.json") as f:
        env = json.load(f)
    return {inst: np.array(env[inst]) for inst in INSTRUMENTS}


# ── Step 4: Predict ───────────────────────────────────────────────
def predict(audio_path):
    print("\n" + "="*50)
    print("  CTFS HARMONIC FINGERPRINT CLASSIFIER")
    print("="*50)
    print(f"\n  Input: {audio_path}")

    # Load audio
    print("  Loading audio...")
    x, sr = load_audio(audio_path)
    duration = len(x) / sr
    print(f"  Duration: {duration:.1f} seconds")

    # Extract features
    print("  Extracting CTFS features...")
    mags = extract_features(x, sr)
    if mags is None or len(mags) == 0:
        print("  ERROR: No valid frames extracted.")
        print("  Make sure audio contains instrument sound, not silence.")
        sys.exit(1)
    print(f"  {len(mags)} valid frames extracted")

    # Compute deviation vectors
    mean_envs   = load_mean_envelopes()
    X_train, y_train = load_training_data()

    # Use global mean for test deviation
    global_mean = np.mean(
        [mean_envs[i] for i in INSTRUMENTS], axis=0
    )
    dev_vectors = mags - global_mean
    X_test      = dev_vectors

    # Scale
    scaler   = StandardScaler()
    X_tr_sc  = scaler.fit_transform(X_train)
    X_te_sc  = scaler.transform(X_test)

    # Train classifiers on full training data
    svm = SVC(kernel='rbf', C=10, gamma='scale',
              probability=True, random_state=42)
    knn = KNeighborsClassifier(n_neighbors=5)
    svm.fit(X_tr_sc, y_train)
    knn.fit(X_tr_sc, y_train)

    # Predict per frame then vote
    svm_preds = svm.predict(X_te_sc)
    knn_preds = knn.predict(X_te_sc)

    # Majority vote
    def majority_vote(preds):
        counts = {inst: np.sum(preds == inst) for inst in INSTRUMENTS}
        return max(counts, key=counts.get), counts

    svm_result, svm_counts = majority_vote(svm_preds)
    knn_result, knn_counts = majority_vote(knn_preds)

    # SVM probabilities
    svm_proba  = svm.predict_proba(X_te_sc)
    mean_proba = np.mean(svm_proba, axis=0)
    classes    = svm.classes_

    # Print results
    print("\n" + "="*50)
    print("  PREDICTION RESULTS")
    print("="*50)
    print(f"\n  SVM  → {svm_result.upper()}")
    print(f"  kNN  → {knn_result.upper()}")

    if svm_result == knn_result:
        print(f"\n  ✅ FINAL PREDICTION: {svm_result.upper()}")
    else:
        print(f"\n  ⚠️  Classifiers disagree!")
        print(f"      Using SVM: {svm_result.upper()}")

    print("\n  Confidence (SVM probabilities):")
    for cls, prob in sorted(zip(classes, mean_proba),
                            key=lambda x: -x[1]):
        bar = "█" * int(prob * 30)
        print(f"    {cls:<8} {prob*100:5.1f}%  {bar}")

    print("\n  Frame vote counts:")
    for inst in INSTRUMENTS:
        s = svm_counts.get(inst, 0)
        k = knn_counts.get(inst, 0)
        print(f"    {inst:<8} SVM:{s:>4}  kNN:{k:>4}")

    print("\n" + "="*50)
    return svm_result


# ── Main ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: py predict.py <audio_file>")
        print("Example: py predict.py test.wav")
        sys.exit(1)

    audio_path = Path(sys.argv[1])
    if not audio_path.exists():
        print(f"File not found: {audio_path}")
        sys.exit(1)

    result = predict(audio_path)