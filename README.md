# CTFS Harmonic Fingerprinting

CTFS Harmonic Fingerprinting is a signal-processing project for identifying musical instruments and studying timbre using **Continuous-Time Fourier Series (CTFS)** harmonics.

The project extracts normalized harmonic fingerprints from short audio frames, compares **sitar**, **veena**, and **guitar** recordings, evaluates reconstruction fidelity, analyzes raga effects, and classifies unknown recordings.

**Repository:** [github.com/prathul-exe/CTFS_Harmonic_Fingerprinting](https://github.com/prathul-exe/CTFS_Harmonic_Fingerprinting)

```bash
git clone https://github.com/prathul-exe/CTFS_Harmonic_Fingerprinting.git
cd CTFS_Harmonic_Fingerprinting
```

---

## Table of Contents

- [Features](#features)
- [Instruments and Ragas](#instruments-and-ragas)
- [How It Works](#how-it-works)
- [Repository Structure](#repository-structure)
- [Installation](#installation)
- [Path Configuration](#path-configuration)
- [Running the Pipeline](#running-the-pipeline)
- [Predict an Audio File](#predict-an-audio-file)
- [Web Application](#web-application)
- [API Endpoints](#api-endpoints)
- [Presentation Demo](#presentation-demo)
- [MATLAB Implementation](#matlab-implementation)
- [Existing Results](#existing-results)
- [Output Files](#output-files)
- [Limitations](#limitations)
- [License](#license)

---

## Features

- 32-harmonic CTFS magnitude and phase extraction
- Fundamental-frequency estimation using autocorrelation
- 100 ms Hann-windowed frames with 50% overlap
- Reconstruction-quality analysis using SNR and minimum harmonic count N*
- Instrument classification using RBF SVM and kNN
- Per-harmonic raga analysis using one-way ANOVA
- PCA, harmonic-profile, heatmap, stem-plot, confusion-matrix, and SNR visualizations
- Flask web interface for audio upload and prediction
- MATLAB implementations of the analysis pipeline

## Instruments and Ragas

| Instrument | Available context |
|------------|-------------------|
| Sitar | Yaman, Bhairavi |
| Veena | Bhairavi, Desh |
| Guitar | Instrument recordings without raga annotations |

Sitar and veena recordings use annotated *pakad* regions. Guitar recordings are processed without raga labels.

---

## How It Works

### 1. Audio Preparation

Recordings are loaded as mono audio at 22,050 Hz. For sitar and veena recordings, `annotate.py` identifies the start and end of the raga pakad region.

### 2. CTFS Extraction

`PYTHON/02_ctfs/compute_ctfs.py`:

- Divides audio into 100 ms frames
- Uses 50% overlap
- Applies a Hann window
- Removes silent frames
- Estimates fundamental frequency using autocorrelation
- Computes the first 32 harmonic coefficients
- Saves harmonic magnitudes and phases as JSON

The harmonic magnitude is normalized by the fundamental:

```text
H(n) = |c_n| / |c_1|
```

This produces a 32-dimensional harmonic envelope representing the instrument's timbre.

### 3. Reconstruction Fidelity

`PYTHON/03_fidelity/fidelity_analysis.py` reconstructs each frame with different numbers of harmonics.

The reconstruction SNR is calculated as:

```text
SNR(N) = 10 log10(signal_power / reconstruction_error_power)
```

**N\*** is the minimum number of harmonics required to reach the **30 dB** SNR threshold.

### 4. Feature Construction

`PYTHON/04_features/build_features.py` calculates the average harmonic envelope for each instrument.

Each feature vector is:

```text
deviation = frame_envelope - instrument_mean_envelope
```

The resulting 32-dimensional vectors are stored with instrument and raga labels.

### 5. Classification

`PYTHON/05_classification/classify.py` evaluates:

- RBF-kernel SVM with `C=10`
- k-nearest neighbors with `k=5`
- Standardized features
- 5-fold stratified cross-validation

The script generates classification metrics and confusion matrices.

### 6. Raga Analysis

`PYTHON/06_analysis/anova_raga_effect.py` performs one-way ANOVA separately for each harmonic to determine whether raga context affects the harmonic fingerprint.

### 7. Visualization

`PYTHON/07_visualize/visualize.py` generates:

- Harmonic profile comparisons
- Deviation heatmaps
- SNR curves
- N* comparison charts
- PCA feature-space plots
- Harmonic stem plots

---

## Repository Structure

```text
.
├── DATA/
│   ├── annotations/       # Pakad timestamp JSON files
│   ├── ctfs/              # Extracted CTFS frames
│   └── features/          # Feature matrix and mean envelopes
├── PYTHON/
│   ├── 02_ctfs/           # CTFS extraction
│   ├── 03_fidelity/       # Reconstruction and SNR analysis
│   ├── 04_features/       # Feature construction
│   ├── 05_classification/ # SVM and kNN classification
│   ├── 06_analysis/       # Raga ANOVA
│   └── 07_visualize/      # Final plots
├── MATLAB/                # MATLAB implementations
├── RESULTS/               # CSV reports and generated figures
├── WEBSITE/
│   ├── app.py             # Flask backend and API
│   └── Static/index.html  # Web interface
├── annotate.py            # Interactive pakad annotation
├── convert.py             # MP3-to-WAV conversion
├── predict.py             # Command-line classifier
├── VIVA_DEMO.py           # Interactive presentation demo
├── Ctfs_Simulation.m      # CTFS MATLAB simulation
└── requirements.txt       # Python dependencies
```

---

## Installation

Install the Python dependencies:

```bash
python -m pip install -r requirements.txt
```

If any packages are missing:

```bash
python -m pip install numpy scipy scikit-learn matplotlib librosa flask sounddevice moviepy
```

MATLAB execution requires MATLAB with the required audio, statistics, signal-processing, and plotting functionality.

## Path Configuration

Several scripts currently use the Windows path:

```text
E:/CTFS_Harmonic_Fingerprinting
```

Before running the project on another computer, update each script's `BASE_DIR` and audio paths to point to the local repository.

Raw audio files are excluded by `.gitignore`, so source recordings must be provided separately.

---

## Running the Pipeline

Run the following commands from the repository root:

```bash
python annotate.py
python PYTHON/02_ctfs/compute_ctfs.py
python PYTHON/03_fidelity/fidelity_analysis.py
python PYTHON/04_features/build_features.py
python PYTHON/05_classification/classify.py
python PYTHON/06_analysis/anova_raga_effect.py
python PYTHON/07_visualize/visualize.py
```

Recommended order:

```text
Annotation
    ↓
CTFS extraction
    ↓
Fidelity analysis
    ↓
Feature construction
    ↓
Classification
    ↓
ANOVA analysis
    ↓
Visualization
```

## Predict an Audio File

After the processed training data has been generated:

```bash
python predict.py path/to/audio.wav
```

The classifier:

- Loads up to 30 seconds of audio
- Extracts valid frames
- Computes 32-harmonic envelopes
- Predicts using SVM and kNN
- Displays frame votes and confidence values

---

## Web Application

The web interface lives in [`WEBSITE/Static/index.html`](WEBSITE/Static/index.html) and is served by the Flask backend in `WEBSITE/app.py`. The page is organized into three sections: **Identify Your Instrument**, **Our Project Results**, and **How Does It Work?**. It uses [Chart.js](https://www.chartjs.org/) (loaded from a CDN) for its charts.

Start the Flask application:

```bash
cd WEBSITE
python app.py
```

Open the application at:

```text
http://localhost:5000
```

The web application supports:

- Audio upload
- Instrument prediction
- SVM probability estimates
- kNN and SVM frame votes
- Harmonic-envelope visualization
- Reconstruction SNR curves
- Precomputed project results

### API Endpoints

`index.html` talks to the backend through these endpoints:

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Serves the web interface |
| `/api/analyze` | POST | Analyzes an uploaded audio file |
| `/api/results` | GET | Returns saved project results |

For `/api/analyze`, upload the audio file using the multipart field name `audio`.

Example:

```bash
curl -X POST -F "audio=@path/to/audio.wav" http://localhost:5000/api/analyze
```

---

## Presentation Demo

`VIVA_DEMO.py` demonstrates:

- Harmonic fingerprints for each instrument
- Sitar reconstruction using different harmonic counts
- Pitch changes while preserving timbre
- Same-pitch comparisons between instruments
- Instrument prediction

Run:

```bash
python VIVA_DEMO.py
```

Audio playback requires a working output device and the `sounddevice` package.

## MATLAB Implementation

The MATLAB pipeline is organized as follows:

```text
MATLAB/02_ctfs/compute_ctfs.m
MATLAB/03_fidelity/fidelity_analysis.m
MATLAB/04_features/build_features.m
MATLAB/05_classification/classify.m
MATLAB/06_analysis/anova_raga_effect.m
MATLAB/07_visualize/visualize.m
```

`Ctfs_Simulation.m` provides an additional standalone CTFS simulation.

---

## Existing Results

### Classification Results

| Classifier | Accuracy | Precision | Recall | F1 Score |
|------------|----------|-----------|--------|----------|
| kNN (k=5) | 96.45% | 96.43% | 96.45% | 96.43% |
| SVM (RBF) | 95.43% | 95.42% | 95.43% | 95.39% |

These results are based on frame-level 5-fold stratified cross-validation.

### Reconstruction Fidelity

| Instrument | N* | SNR at N* | Valid Frames |
|------------|----|-----------|--------------|
| Sitar | 9 | 30.37 dB | 792 |
| Veena | 9 | 30.59 dB | 752 |
| Guitar | 13 | 30.41 dB | 2486 |

### Raga Analysis

The ANOVA output is stored in:

```text
RESULTS/anova_results.csv
```

> **Note:** The current generated report contains `NaN` values for aggregate F-statistics and p-values. Rerun the ANOVA script and verify the regenerated numeric output before drawing conclusions.

---

## Output Files

| Output | Location |
|--------|----------|
| CTFS frame data | `DATA/ctfs/*.json` |
| Mean harmonic envelopes | `DATA/features/mean_envelopes.json` |
| Feature matrix | `DATA/features/feature_matrix.csv` |
| Classification report | `RESULTS/classification_report.csv` |
| N* report | `RESULTS/N_star_table.csv` |
| ANOVA report | `RESULTS/anova_results.csv` |
| Generated plots | `RESULTS/figures/` |

## Limitations

- Raw audio recordings are not included.
- Several scripts use hard-coded Windows paths.
- Frame-level cross-validation may contain frames from the same recording in both training and validation folds.
- The system only recognizes three instrument classes.
- Raga analysis is limited to the available sitar and veena recordings.
- N* is based on reconstruction SNR and not perceptual listening tests.
- ANOVA results should be regenerated before interpretation.
