"""
app.py - Flask backend for CTFS Harmonic Fingerprinting Web App
Save this in E:/CTFS_Harmonic_Fingerprinting/website/app.py
Run: py app.py
Open: http://localhost:5000
"""

from flask import Flask, request, jsonify, send_from_directory
import numpy as np
import json
import os
import csv
from pathlib import Path
import librosa
import tempfile

app = Flask(__name__, static_folder='static', static_url_path='')

BASE_DIR  = Path("E:/CTFS_Harmonic_Fingerprinting")
FEAT_DIR  = BASE_DIR / "data" / "features"
CTFS_DIR  = BASE_DIR / "data" / "ctfs"

N_HARMONICS   = 32
FRAME_MS      = 100
TARGET_SR     = 22050
ENERGY_THRESH = 0.01
INSTRUMENTS   = ["sitar", "veena", "guitar"]
MAX_DURATION  = 30.0


# ── Audio processing helpers ──────────────────────────────────────
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
    all_mags, all_f0s = [], []
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
        all_f0s.append(f0)
    return (np.array(all_mags), np.array(all_f0s)) if all_mags else (None, None)


def compute_snr_curve(mags_arr, f0s_arr, n_samples=50):
    """Compute mean SNR curve over n_samples frames."""
    frame_len = int(FRAME_MS * 1e-3 * TARGET_SR)
    t         = np.arange(frame_len) / TARGET_SR
    indices   = np.random.choice(len(mags_arr),
                                  min(n_samples, len(mags_arr)),
                                  replace=False)
    snr_matrix = []
    for idx in indices:
        mags = mags_arr[idx]
        f0   = f0s_arr[idx]
        # Full reconstruction
        original = sum(mags[n] * np.cos((n+1) * 2 * np.pi * f0 * t)
                       for n in range(N_HARMONICS))
        snr_row = []
        for N in range(1, N_HARMONICS+1):
            recon = sum(mags[n] * np.cos((n+1) * 2 * np.pi * f0 * t)
                        for n in range(N))
            err   = original - recon
            sp    = np.sum(original**2)
            ep    = np.sum(err**2)
            snr   = 10 * np.log10(sp / ep) if ep > 1e-12 else 100.0
            snr_row.append(float(snr))
        snr_matrix.append(snr_row)
    mean_snr = np.mean(snr_matrix, axis=0).tolist()
    # Find N*
    n_star = N_HARMONICS
    for i, s in enumerate(mean_snr):
        if s >= 30.0:
            n_star = i + 1
            break
    return mean_snr, n_star


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


def predict_instrument(mags_arr):
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.svm import SVC
    from sklearn.preprocessing import StandardScaler

    X_train, y_train = load_training_data()
    mean_envs        = load_mean_envelopes()

    # Reconstruct raw magnitudes
    X_train_raw = []
    for feat, label in zip(X_train, y_train):
        X_train_raw.append(feat + mean_envs[label])
    X_train_raw = np.array(X_train_raw)

    scaler   = StandardScaler()
    X_tr_sc  = scaler.fit_transform(X_train_raw)
    X_te_sc  = scaler.transform(mags_arr)

    svm = SVC(kernel='rbf', C=10, gamma='scale', probability=True, random_state=42)
    knn = KNeighborsClassifier(n_neighbors=5)
    svm.fit(X_tr_sc, y_train)
    knn.fit(X_tr_sc, y_train)

    svm_preds  = svm.predict(X_te_sc)
    knn_preds  = knn.predict(X_te_sc)
    svm_proba  = svm.predict_proba(X_te_sc)
    mean_proba = np.mean(svm_proba, axis=0)
    classes    = list(svm.classes_)

    def vote(preds):
        counts = {i: int(np.sum(preds == i)) for i in INSTRUMENTS}
        return max(counts, key=counts.get), counts

    knn_result, knn_counts = vote(knn_preds)
    svm_result, svm_counts = vote(svm_preds)

    proba_dict = {cls: float(p) for cls, p in zip(classes, mean_proba)}

    return {
        "prediction"  : knn_result,
        "svm_result"  : svm_result,
        "knn_counts"  : knn_counts,
        "svm_counts"  : svm_counts,
        "probabilities": proba_dict,
        "n_frames"    : len(mags_arr),
    }


# ── Routes ────────────────────────────────────────────────────────
@app.route('/')
def index():
    return send_from_directory('static', 'index.html')


@app.route('/api/analyze', methods=['POST'])
def analyze():
    if 'audio' not in request.files:
        return jsonify({'error': 'No audio file'}), 400

    audio_file = request.files['audio']

    # Save to temp file
    suffix = os.path.splitext(audio_file.filename)[1] or '.wav'
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        audio_file.save(tmp.name)
        tmp_path = tmp.name

    try:
        # Load audio
        x, sr = librosa.load(tmp_path, sr=TARGET_SR, mono=True,
                              duration=MAX_DURATION)
        x = x / (np.max(np.abs(x)) + 1e-9)
        duration = len(x) / sr

        # Extract features
        mags_arr, f0s_arr = extract_features(x, sr)
        if mags_arr is None:
            return jsonify({'error': 'No valid frames found. Make sure audio contains instrument sound.'}), 400

        # Predict instrument
        prediction = predict_instrument(mags_arr)

        # Compute SNR curve (use fewer samples for speed)
        mean_snr, n_star = compute_snr_curve(mags_arr, f0s_arr, n_samples=20)

        # Mean harmonic envelope of input
        mean_envelope = np.mean(mags_arr, axis=0).tolist()

        # Load reference envelopes for comparison
        ref_envelopes = {}
        for inst in INSTRUMENTS:
            ctfs_path = CTFS_DIR / f"{inst}_ctfs.json"
            if ctfs_path.exists():
                with open(ctfs_path) as f:
                    data = json.load(f)
                ref_mags = np.array([d["magnitudes"] for d in data])
                ref_envelopes[inst] = np.mean(ref_mags, axis=0).tolist()

        return jsonify({
            'duration'      : round(duration, 1),
            'n_frames'      : int(len(mags_arr)),
            'prediction'    : prediction,
            'snr_curve'     : mean_snr,
            'n_star'        : int(n_star),
            'mean_envelope' : mean_envelope,
            'ref_envelopes' : ref_envelopes,
        })

    finally:
        os.unlink(tmp_path)


@app.route('/api/results')
def results():
    """Return pre-computed project results for the Results tab."""
    # Load classification report
    clf_report = {}
    clf_path = BASE_DIR / "results" / "classification_report.csv"
    if clf_path.exists():
        with open(clf_path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                clf_report[row['classifier']] = row

    # Load N* table
    nstar_table = {}
    nstar_path = BASE_DIR / "results" / "N_star_table.csv"
    if nstar_path.exists():
        with open(nstar_path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                nstar_table[row['instrument']] = row

    # Load ANOVA results
    anova_results = {}
    anova_path = BASE_DIR / "results" / "anova_results.csv"
    if anova_path.exists():
        with open(anova_path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                anova_results[row['instrument']] = row

    # Load reference SNR curves from ctfs data
    ref_snr = {}
    for inst in INSTRUMENTS:
        ctfs_path = CTFS_DIR / f"{inst}_ctfs.json"
        if ctfs_path.exists():
            with open(ctfs_path) as f:
                data = json.load(f)
            mags_arr = np.array([d["magnitudes"] for d in data[:30]])
            f0s_arr  = np.array([d["f0_hz"] for d in data[:30]])
            snr, ns  = compute_snr_curve(mags_arr, f0s_arr, n_samples=20)
            ref_snr[inst] = {'snr': snr, 'n_star': ns}

    # Reference harmonic envelopes
    ref_envelopes = {}
    for inst in INSTRUMENTS:
        ctfs_path = CTFS_DIR / f"{inst}_ctfs.json"
        if ctfs_path.exists():
            with open(ctfs_path) as f:
                data = json.load(f)
            mags = np.array([d["magnitudes"] for d in data])
            ref_envelopes[inst] = np.mean(mags, axis=0).tolist()

    return jsonify({
        'classification' : clf_report,
        'nstar'          : nstar_table,
        'anova'          : anova_results,
        'ref_snr'        : ref_snr,
        'ref_envelopes'  : ref_envelopes,
    })


if __name__ == '__main__':
    print("\n" + "="*50)
    print("  CTFS Harmonic Fingerprinting — Web App")
    print("="*50)
    print("  Open your browser and go to:")
    print("  http://localhost:5000")
    print("="*50 + "\n")
    app.run(debug=True, port=5000)
