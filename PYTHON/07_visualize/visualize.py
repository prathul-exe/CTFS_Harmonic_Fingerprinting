"""
visualize.py
------------
Module 7 — Final Visualization

Generates all publication-quality plots:
1. Harmonic profile comparison (all 3 instruments)
2. Deviation heatmap per instrument
3. SNR curves (from Module 3)
4. Confusion matrices (from Module 5)
5. ANOVA F-statistic plots (from Module 6)
6. N* threshold comparison bar chart
7. Feature space scatter (PCA 2D)

Usage:
    python python/07_visualize/visualize.py
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import csv

# ── Parameters ────────────────────────────────────────────────────
BASE_DIR    = Path("E:/CTFS_Harmonic_Fingerprinting")
CTFS_DIR    = BASE_DIR / "data" / "ctfs"
FEAT_DIR    = BASE_DIR / "data" / "features"
RESULTS_DIR = BASE_DIR / "results"
FIGURES_DIR = RESULTS_DIR / "figures"
N_HARMONICS = 32
BG          = "#1a1a2e"
INSTRUMENTS = ["sitar", "veena", "guitar"]
COLORS      = {"sitar": "#B85042", "veena": "#A7BEAE", "guitar": "#E7E8D1"}


# ── Plot 1: Harmonic Profile Comparison ───────────────────────────
def plot_harmonic_profiles():
    print("  [1] Harmonic profile comparison...")
    fig, ax = plt.subplots(figsize=(11, 6))
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)

    n_vals = np.arange(1, N_HARMONICS+1)

    for inst in INSTRUMENTS:
        path = CTFS_DIR / f"{inst}_ctfs.json"
        with open(path) as f:
            data = json.load(f)
        mags    = np.array([d["magnitudes"] for d in data])
        mean_m  = np.mean(mags, axis=0)
        std_m   = np.std(mags, axis=0)
        color   = COLORS[inst]

        ax.plot(n_vals, mean_m, color=color, linewidth=2.5,
                label=inst.capitalize())
        ax.fill_between(n_vals, mean_m-std_m, mean_m+std_m,
                        alpha=0.15, color=color)

    ax.set_xlabel("Harmonic Index (n)", color='white', fontsize=12)
    ax.set_ylabel("Normalized |cₙ|",   color='white', fontsize=12)
    ax.set_title("Harmonic Fingerprint Comparison — Sitar vs Veena vs Guitar",
                 color='white', fontsize=13, fontweight='bold')
    ax.legend(facecolor='#2a2a3e', labelcolor='white', fontsize=11)
    ax.tick_params(colors='white')
    for s in ['bottom','left']: ax.spines[s].set_color('white')
    for s in ['top','right']:   ax.spines[s].set_visible(False)
    ax.grid(True, alpha=0.15, color='white')

    plt.tight_layout()
    out = FIGURES_DIR / "harmonic_profiles.png"
    plt.savefig(str(out), dpi=150, bbox_inches='tight', facecolor=BG)
    plt.close()
    print(f"     Saved: {out}")


# ── Plot 2: Deviation Heatmap ─────────────────────────────────────
def plot_deviation_heatmap():
    print("  [2] Deviation heatmap...")
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.patch.set_facecolor(BG)

    for ax, inst in zip(axes, INSTRUMENTS):
        path = CTFS_DIR / f"{inst}_ctfs.json"
        with open(path) as f:
            data = json.load(f)
        mags     = np.array([d["magnitudes"] for d in data])
        mean_env = np.mean(mags, axis=0)
        dev      = mags - mean_env

        # Sample max 200 frames for readability
        sample = dev[:200] if len(dev) > 200 else dev

        im = ax.imshow(sample.T, aspect='auto', cmap='RdBu_r',
                       vmin=-0.5, vmax=0.5)
        ax.set_facecolor(BG)
        ax.set_xlabel("Frame index", color='white', fontsize=10)
        ax.set_ylabel("Harmonic n",  color='white', fontsize=10)
        ax.set_title(inst.capitalize(), color=COLORS[inst],
                     fontsize=12, fontweight='bold')
        ax.tick_params(colors='white')
        for s in ax.spines.values(): s.set_edgecolor('white')

    plt.colorbar(im, ax=axes[-1], label='Deviation').ax.yaxis.set_tick_params(color='white')
    plt.suptitle("CTFS Deviation Heatmaps",
                 color='white', fontsize=14, fontweight='bold')
    plt.tight_layout()
    out = FIGURES_DIR / "deviation_heatmaps.png"
    plt.savefig(str(out), dpi=150, bbox_inches='tight', facecolor=BG)
    plt.close()
    print(f"     Saved: {out}")


# ── Plot 3: N* Comparison Bar Chart ───────────────────────────────
def plot_N_star_comparison():
    print("  [3] N* threshold comparison...")
    csv_path = RESULTS_DIR / "N_star_table.csv"
    insts, n_stars = [], []
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            insts.append(row["instrument"])
            n_stars.append(int(row["N_star"]))

    fig, ax = plt.subplots(figsize=(8, 5))
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)

    bar_colors = [COLORS.get(i, 'white') for i in insts]
    bars = ax.bar(insts, n_stars, color=bar_colors, alpha=0.85,
                  width=0.5, edgecolor='white', linewidth=0.5)

    for bar, val in zip(bars, n_stars):
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + 0.3,
                f"N*={val}", ha='center', va='bottom',
                color='white', fontsize=12, fontweight='bold')

    ax.set_xlabel("Instrument", color='white', fontsize=12)
    ax.set_ylabel("N* (harmonics for 30dB SNR)", color='white', fontsize=12)
    ax.set_title("Minimum Harmonic Threshold N* per Instrument",
                 color='white', fontsize=13, fontweight='bold')
    ax.tick_params(colors='white')
    ax.set_xticklabels([i.capitalize() for i in insts], color='white')
    for s in ['bottom','left']: ax.spines[s].set_color('white')
    for s in ['top','right']:   ax.spines[s].set_visible(False)
    ax.grid(True, alpha=0.15, color='white', axis='y')
    ax.set_ylim(0, max(n_stars) + 4)

    plt.tight_layout()
    out = FIGURES_DIR / "N_star_comparison.png"
    plt.savefig(str(out), dpi=150, bbox_inches='tight', facecolor=BG)
    plt.close()
    print(f"     Saved: {out}")


# ── Plot 4: PCA Feature Space ─────────────────────────────────────
def plot_pca_feature_space():
    print("  [4] PCA feature space scatter...")

    feat_path = FEAT_DIR / "feature_matrix.csv"
    X, y = [], []
    with open(feat_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            X.append([float(row[f"h{n}"]) for n in range(1, 33)])
            y.append(row["instrument"])
    X = np.array(X)
    y = np.array(y)

    # PCA to 2D
    scaler = StandardScaler()
    X_s    = scaler.fit_transform(X)
    pca    = PCA(n_components=2)
    X_2d   = pca.fit_transform(X_s)

    fig, ax = plt.subplots(figsize=(9, 7))
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)

    for inst in INSTRUMENTS:
        mask = y == inst
        ax.scatter(X_2d[mask, 0], X_2d[mask, 1],
                   c=COLORS[inst], label=inst.capitalize(),
                   alpha=0.4, s=8)

    var = pca.explained_variance_ratio_
    ax.set_xlabel(f"PC1 ({var[0]*100:.1f}% variance)",
                  color='white', fontsize=11)
    ax.set_ylabel(f"PC2 ({var[1]*100:.1f}% variance)",
                  color='white', fontsize=11)
    ax.set_title("PCA — CTFS Feature Space (2D Projection)",
                 color='white', fontsize=13, fontweight='bold')
    ax.legend(facecolor='#2a2a3e', labelcolor='white',
              fontsize=11, markerscale=3)
    ax.tick_params(colors='white')
    for s in ['bottom','left']: ax.spines[s].set_color('white')
    for s in ['top','right']:   ax.spines[s].set_visible(False)
    ax.grid(True, alpha=0.15, color='white')

    plt.tight_layout()
    out = FIGURES_DIR / "pca_feature_space.png"
    plt.savefig(str(out), dpi=150, bbox_inches='tight', facecolor=BG)
    plt.close()
    print(f"     Saved: {out}")


# ── Plot 5: Stem Plot (single note comparison) ────────────────────
def plot_stem_comparison():
    print("  [5] Stem plot comparison...")

    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)
    fig.patch.set_facecolor(BG)

    n_vals = np.arange(1, N_HARMONICS+1)

    for ax, inst in zip(axes, INSTRUMENTS):
        path = CTFS_DIR / f"{inst}_ctfs.json"
        with open(path) as f:
            data = json.load(f)
        mags    = np.array([d["magnitudes"] for d in data])
        mean_m  = np.mean(mags, axis=0)
        color   = COLORS[inst]

        ax.set_facecolor(BG)
        markerline, stemlines, baseline = ax.stem(
            n_vals, mean_m, linefmt='-', markerfmt='o', basefmt=' '
        )
        plt.setp(stemlines,   color=color, linewidth=1.5, alpha=0.8)
        plt.setp(markerline,  color=color, markersize=5)

        ax.set_xlabel("Harmonic n", color='white', fontsize=11)
        ax.set_ylabel("Normalized |cₙ|" if inst == "sitar" else "",
                      color='white', fontsize=11)
        ax.set_title(inst.capitalize(), color=color,
                     fontsize=13, fontweight='bold')
        ax.tick_params(colors='white')
        for s in ['bottom','left']: ax.spines[s].set_color('white')
        for s in ['top','right']:   ax.spines[s].set_visible(False)
        ax.grid(True, alpha=0.15, color='white')

    plt.suptitle("CTFS Harmonic Stem Plots — Mean Envelope per Instrument",
                 color='white', fontsize=14, fontweight='bold')
    plt.tight_layout()
    out = FIGURES_DIR / "stem_plots.png"
    plt.savefig(str(out), dpi=150, bbox_inches='tight', facecolor=BG)
    plt.close()
    print(f"     Saved: {out}")


# ── Main ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("MODULE 7 — FINAL VISUALIZATION")
    print("="*45)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    plot_harmonic_profiles()
    plot_deviation_heatmap()
    plot_N_star_comparison()
    plot_pca_feature_space()
    plot_stem_comparison()

    print("\n" + "="*45)
    print("ALL PLOTS GENERATED")
    print(f"Check: {FIGURES_DIR}")
    print("="*45)
    print("\nFinal figures:")
    for f in sorted(FIGURES_DIR.glob("*.png")):
        print(f"  {f.name}")
        