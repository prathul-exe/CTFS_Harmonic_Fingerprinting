"""
classify.py
-----------
Module 5 — SVM + kNN Classification

1. Load feature matrix from Module 4
2. Train SVM (RBF kernel) + kNN (k=5)
3. 5-fold stratified cross validation
4. Report accuracy, precision, recall, F1
5. Plot confusion matrices
6. Compare with MFCC baseline

Usage:
    python python/05_classification/classify.py
"""

import json
import numpy as np
import csv
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import (classification_report, confusion_matrix,
                             accuracy_score)
from sklearn.preprocessing import LabelEncoder, StandardScaler

# ── Parameters ────────────────────────────────────────────────────
BASE_DIR    = Path("E:/CTFS_Harmonic_Fingerprinting")
FEAT_DIR    = BASE_DIR / "data" / "features"
RESULTS_DIR = BASE_DIR / "results"
FIGURES_DIR = RESULTS_DIR / "figures"

INSTRUMENTS = ["sitar", "veena", "guitar"]
N_FOLDS     = 5
BG          = "#1a1a2e"
COLORS      = ["#B85042", "#A7BEAE", "#E7E8D1"]


# ── Load feature matrix ───────────────────────────────────────────
def load_features():
    feat_path = FEAT_DIR / "feature_matrix.csv"
    X, y, ragas = [], [], []
    with open(feat_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            feats = [float(row[f"h{n}"]) for n in range(1, 33)]
            X.append(feats)
            y.append(row["instrument"])
            ragas.append(row["raga"])
    return np.array(X), np.array(y), np.array(ragas)


# ── Plot confusion matrix ─────────────────────────────────────────
def plot_confusion_matrix(cm, title, filename):
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 6))
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)

    im = ax.imshow(cm, interpolation='nearest', cmap='Reds')
    plt.colorbar(im, ax=ax).ax.yaxis.set_tick_params(color='white')

    ax.set_xticks(range(len(INSTRUMENTS)))
    ax.set_yticks(range(len(INSTRUMENTS)))
    ax.set_xticklabels([i.capitalize() for i in INSTRUMENTS],
                       color='white', fontsize=11)
    ax.set_yticklabels([i.capitalize() for i in INSTRUMENTS],
                       color='white', fontsize=11)

    # Annotate cells
    thresh = cm.max() / 2
    for i in range(len(INSTRUMENTS)):
        for j in range(len(INSTRUMENTS)):
            color = 'white' if cm[i,j] < thresh else 'black'
            ax.text(j, i, str(cm[i,j]), ha='center', va='center',
                    color=color, fontsize=13, fontweight='bold')

    ax.set_xlabel("Predicted", color='white', fontsize=12)
    ax.set_ylabel("True", color='white', fontsize=12)
    ax.set_title(title, color='white', fontsize=13, fontweight='bold')
    ax.tick_params(colors='white')
    for spine in ax.spines.values():
        spine.set_edgecolor('white')

    plt.tight_layout()
    out = FIGURES_DIR / filename
    plt.savefig(str(out), dpi=150, bbox_inches='tight', facecolor=BG)
    plt.close()
    print(f"  Plot saved: {out}")


# ── Run classifier ────────────────────────────────────────────────
def run_classifier(clf, X, y, name):
    print(f"\n  [{name}]")
    le    = LabelEncoder()
    y_enc = le.fit_transform(y)
    skf   = StratifiedKFold(n_splits=N_FOLDS, shuffle=True,
                            random_state=42)

    y_pred = cross_val_predict(clf, X, y_enc, cv=skf)
    y_pred_labels = le.inverse_transform(y_pred)

    acc = accuracy_score(y, y_pred_labels)
    print(f"  Accuracy : {acc*100:.2f}%")

    report = classification_report(y, y_pred_labels,
                                   target_names=INSTRUMENTS,
                                   output_dict=True)
    print(f"  Precision: {report['weighted avg']['precision']*100:.2f}%")
    print(f"  Recall   : {report['weighted avg']['recall']*100:.2f}%")
    print(f"  F1 Score : {report['weighted avg']['f1-score']*100:.2f}%")

    cm = confusion_matrix(y, y_pred_labels, labels=INSTRUMENTS)
    return acc, report, cm, y_pred_labels


# ── Save classification report ────────────────────────────────────
def save_report(results):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = RESULTS_DIR / "classification_report.csv"
    with open(out, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["classifier", "accuracy", "precision",
                         "recall", "f1"])
        for name, acc, rep in results:
            wa = rep["weighted avg"]
            writer.writerow([name,
                             f"{acc*100:.2f}",
                             f"{wa['precision']*100:.2f}",
                             f"{wa['recall']*100:.2f}",
                             f"{wa['f1-score']*100:.2f}"])
    print(f"\n  Report saved: {out}")


# ── Main ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("MODULE 5 — CLASSIFICATION")
    print("="*45)

    # Load data
    print("\nLoading feature matrix...")
    X, y, ragas = load_features()
    print(f"  X shape : {X.shape}")
    print(f"  Classes : {np.unique(y)}")

    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Define classifiers
    classifiers = [
        (SVC(kernel='rbf', C=10, gamma='scale',
             probability=True, random_state=42),
         "SVM (RBF)"),
        (KNeighborsClassifier(n_neighbors=5),
         "kNN (k=5)"),
    ]

    all_results = []
    print("\nRunning 5-fold stratified cross-validation...")

    for clf, name in classifiers:
        acc, report, cm, _ = run_classifier(clf, X_scaled, y, name)
        all_results.append((name, acc, report))
        fname = f"confusion_matrix_{name.replace(' ','_').replace('(','').replace(')','')}.png"
        plot_confusion_matrix(cm, f"Confusion Matrix — {name}", fname)

    # Summary table
    print("\n" + "="*45)
    print("  RESULTS SUMMARY")
    print("="*45)
    print(f"  {'Classifier':<15} {'Accuracy':>10} {'F1':>10}")
    print("  " + "-"*35)
    for name, acc, rep in all_results:
        f1 = rep['weighted avg']['f1-score']
        print(f"  {name:<15} {acc*100:>9.2f}% {f1*100:>9.2f}%")

    save_report(all_results)

    print("\n" + "="*45)
    print("MODULE 5 COMPLETE")
    print("Check results/figures/ for confusion matrices")
    print("Proceed to Module 6 — ANOVA Raga Effect")
    print("="*45)