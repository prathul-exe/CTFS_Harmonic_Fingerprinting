"""
anova_raga_effect.py
--------------------
Module 6 — ANOVA Raga Effect Analysis

Tests whether raga context significantly shifts the
harmonic fingerprint of the same instrument.

For each instrument (sitar, veena):
1. Load CTFS features grouped by raga
2. Run one-way ANOVA across raga groups
3. Report F-statistic and p-value
4. Plot mean harmonic profiles per raga
5. Save results

Usage:
    python python/06_analysis/anova_raga_effect.py
"""

import json
import numpy as np
import csv
import matplotlib.pyplot as plt
from pathlib import Path
from scipy import stats

# ── Parameters ────────────────────────────────────────────────────
BASE_DIR    = Path("E:/CTFS_Harmonic_Fingerprinting")
CTFS_DIR    = BASE_DIR / "data" / "ctfs"
RESULTS_DIR = BASE_DIR / "results"
FIGURES_DIR = RESULTS_DIR / "figures"
N_HARMONICS = 32
BG          = "#1a1a2e"

# Instruments with multiple ragas
RAGA_MAP = {
    "sitar" : ["yaman", "bhairavi"],
    "veena" : ["bhairavi", "desh"],
}


# ── Load CTFS grouped by raga ─────────────────────────────────────
def load_by_raga(instrument):
    path = CTFS_DIR / f"{instrument}_ctfs.json"
    with open(path) as f:
        data = json.load(f)

    raga_groups = {}
    for frame in data:
        raga = frame["raga"]
        mags = np.array(frame["magnitudes"])
        if raga not in raga_groups:
            raga_groups[raga] = []
        raga_groups[raga].append(mags)

    # Convert to arrays
    for raga in raga_groups:
        raga_groups[raga] = np.array(raga_groups[raga])

    return raga_groups


# ── Run ANOVA per harmonic ────────────────────────────────────────
def run_anova(instrument, raga_groups):
    """
    For each harmonic n, run one-way ANOVA across raga groups.
    Tests: does raga significantly affect harmonic n?

    Returns:
        f_stats : F-statistic per harmonic
        p_vals  : p-value per harmonic
    """
    ragas    = list(raga_groups.keys())
    f_stats  = np.zeros(N_HARMONICS)
    p_vals   = np.zeros(N_HARMONICS)

    for n in range(N_HARMONICS):
        groups = [raga_groups[r][:, n] for r in ragas]
        f, p   = stats.f_oneway(*groups)
        f_stats[n] = f
        p_vals[n]  = p

    # Overall significance
    sig_harmonics = np.sum(p_vals < 0.05)
    print(f"  Significant harmonics (p<0.05): "
          f"{sig_harmonics}/{N_HARMONICS}")

    mean_f = np.mean(f_stats)
    min_p  = np.min(p_vals)
    print(f"  Mean F-statistic : {mean_f:.2f}")
    print(f"  Min p-value      : {min_p:.6f}")

    if min_p < 0.05:
        print(f"  ✅ Raga SIGNIFICANTLY affects harmonic fingerprint")
    else:
        print(f"  ℹ️  No significant raga effect detected")

    return f_stats, p_vals


# ── Plot harmonic profiles per raga ───────────────────────────────
def plot_raga_profiles(instrument, raga_groups):
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    raga_colors = {
        "yaman"    : "#B85042",
        "bhairavi" : "#A7BEAE",
        "desh"     : "#E7E8D1",
    }

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.patch.set_facecolor(BG)

    n_vals = np.arange(1, N_HARMONICS+1)

    # Left: mean harmonic profiles per raga
    ax1 = axes[0]
    ax1.set_facecolor(BG)
    for raga, mags in raga_groups.items():
        mean_mag = np.mean(mags, axis=0)
        std_mag  = np.std(mags, axis=0)
        color    = raga_colors.get(raga, "white")
        ax1.plot(n_vals, mean_mag,
                 label=f"Raga {raga.capitalize()}",
                 color=color, linewidth=2.5)
        ax1.fill_between(n_vals,
                         mean_mag - std_mag,
                         mean_mag + std_mag,
                         alpha=0.15, color=color)

    ax1.set_xlabel("Harmonic Index (n)", color='white', fontsize=11)
    ax1.set_ylabel("Normalized |cₙ|", color='white', fontsize=11)
    ax1.set_title(f"{instrument.capitalize()} — Harmonic Profile by Raga",
                  color='white', fontsize=12, fontweight='bold')
    ax1.legend(facecolor='#2a2a3e', labelcolor='white', fontsize=10)
    ax1.tick_params(colors='white')
    ax1.spines['bottom'].set_color('white')
    ax1.spines['left'].set_color('white')
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.grid(True, alpha=0.15, color='white')

    # Right: F-statistic per harmonic
    ax2 = axes[1]
    ax2.set_facecolor(BG)
    f_stats, p_vals = run_anova_silent(instrument, raga_groups)
    bar_colors = ['#B85042' if p < 0.05 else '#555577'
                  for p in p_vals]
    ax2.bar(n_vals, f_stats, color=bar_colors, alpha=0.85)
    ax2.axhline(y=3.84, color='white', linestyle='--',
                linewidth=1.5, label='F critical (p=0.05)')
    ax2.set_xlabel("Harmonic Index (n)", color='white', fontsize=11)
    ax2.set_ylabel("F-statistic", color='white', fontsize=11)
    ax2.set_title(f"{instrument.capitalize()} — ANOVA F-statistic per Harmonic",
                  color='white', fontsize=12, fontweight='bold')
    ax2.legend(facecolor='#2a2a3e', labelcolor='white', fontsize=10)
    ax2.tick_params(colors='white')
    ax2.spines['bottom'].set_color('white')
    ax2.spines['left'].set_color('white')
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.grid(True, alpha=0.15, color='white')

    # Red bars = significant
    from matplotlib.patches import Patch
    legend_els = [
        Patch(facecolor='#B85042', label='Significant (p<0.05)'),
        Patch(facecolor='#555577', label='Not significant'),
    ]
    ax2.legend(handles=legend_els, facecolor='#2a2a3e',
               labelcolor='white', fontsize=10)

    plt.suptitle(f"Raga Effect Analysis — {instrument.capitalize()}",
                 color='white', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()

    out = FIGURES_DIR / f"anova_{instrument}.png"
    plt.savefig(str(out), dpi=150, bbox_inches='tight', facecolor=BG)
    plt.close()
    print(f"  Plot saved: {out}")


def run_anova_silent(instrument, raga_groups):
    ragas   = list(raga_groups.keys())
    f_stats = np.zeros(N_HARMONICS)
    p_vals  = np.zeros(N_HARMONICS)
    for n in range(N_HARMONICS):
        groups = [raga_groups[r][:, n] for r in ragas]
        f, p   = stats.f_oneway(*groups)
        f_stats[n] = f
        p_vals[n]  = p
    return f_stats, p_vals


# ── Save ANOVA results ────────────────────────────────────────────
def save_anova_results(all_results):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = RESULTS_DIR / "anova_results.csv"
    with open(out, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["instrument", "mean_F", "min_p",
                         "sig_harmonics", "significant"])
        for row in all_results:
            writer.writerow(row)
    print(f"\n  ANOVA results saved: {out}")


# ── Main ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("MODULE 6 — ANOVA RAGA EFFECT ANALYSIS")
    print("="*45)

    all_results = []

    for inst, ragas in RAGA_MAP.items():
        print(f"\n[{inst.upper()}] — Ragas: {ragas}")
        raga_groups = load_by_raga(inst)

        # Only keep ragas in our map
        raga_groups = {r: raga_groups[r] for r in ragas
                       if r in raga_groups}

        print(f"  Frames per raga:")
        for r, mags in raga_groups.items():
            print(f"    {r}: {len(mags)} frames")

        f_stats, p_vals = run_anova(inst, raga_groups)
        plot_raga_profiles(inst, raga_groups)

        sig = np.sum(p_vals < 0.05)
        all_results.append([
            inst,
            f"{np.mean(f_stats):.2f}",
            f"{np.min(p_vals):.6f}",
            sig,
            "YES" if np.min(p_vals) < 0.05 else "NO"
        ])

    save_anova_results(all_results)

    print("\n" + "="*45)
    print("  ANOVA SUMMARY")
    print("="*45)
    print(f"  {'Instrument':<10} {'Sig. Harmonics':>15} {'Result':>10}")
    print("  " + "-"*38)
    for row in all_results:
        print(f"  {row[0]:<10} {str(row[3])+'/32':>15} {row[4]:>10}")

    print("\n" + "="*45)
    print("MODULE 6 COMPLETE")
    print("Proceed to Module 7 — Visualization")
    print("="*45)