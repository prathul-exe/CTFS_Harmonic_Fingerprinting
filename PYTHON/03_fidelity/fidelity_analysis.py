"""
fidelity_analysis.py
--------------------
Module 3 — Reconstruction Fidelity Analysis

For each instrument:
1. Load CTFS coefficients from Module 2 output
2. Reconstruct signal using N=1,2,...,32 harmonics
3. Compute SNR(N) at each step
4. Find N* where SNR >= 30dB threshold
5. Plot SNR curves and save N* table

Usage:
    python python/03_fidelity/fidelity_analysis.py
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# ── Parameters ────────────────────────────────────────────────────
N_HARMONICS  = 32
SNR_THRESHOLD = 30.0      # dB — minimum acceptable fidelity
FRAME_MS     = 100
TARGET_SR    = 22050

BASE_DIR     = Path("E:/CTFS_Harmonic_Fingerprinting")
CTFS_DIR     = BASE_DIR / "data" / "ctfs"
RESULTS_DIR  = BASE_DIR / "results"
FIGURES_DIR  = RESULTS_DIR / "figures"

INSTRUMENTS  = ["sitar", "veena", "guitar"]


# ── Step 1: Reconstruct signal from N harmonics ───────────────────
def reconstruct_signal(magnitudes, phases, f0, N, sr):
    """
    Reconstruct signal using first N CTFS harmonics.

    x̂_N(t) = Σ_{n=1}^{N} |c_n| · cos(n·ω₀·t + ∠c_n)

    Args:
        magnitudes : array of shape (N_HARMONICS,)
        phases     : array of shape (N_HARMONICS,)
        f0         : fundamental frequency in Hz
        N          : number of harmonics to use
        sr         : sample rate

    Returns:
        x_hat : reconstructed signal array
    """
    frame_len = int(FRAME_MS * 1e-3 * sr)
    t         = np.arange(frame_len) / sr
    omega0    = 2 * np.pi * f0
    x_hat     = np.zeros(frame_len)

    for n in range(1, N+1):
        x_hat += magnitudes[n-1] * np.cos(n * omega0 * t + phases[n-1])

    return x_hat


# ── Step 2: Compute SNR ───────────────────────────────────────────
def compute_snr(original, reconstructed):
    """
    SNR(N) = 10 · log10( ||x||² / ||x - x̂_N||² )

    Higher SNR = better reconstruction.
    """
    signal_power = np.sum(original ** 2)
    error_power  = np.sum((original - reconstructed) ** 2)

    if error_power < 1e-12:
        return 100.0   # perfect reconstruction

    snr = 10 * np.log10(signal_power / error_power)
    return snr


# ── Step 3: Find N* ───────────────────────────────────────────────
def find_N_star(snr_curve, threshold=SNR_THRESHOLD):
    """
    Find minimum N where SNR >= threshold.
    Returns N* or N_HARMONICS if threshold never reached.
    """
    for n, snr in enumerate(snr_curve, start=1):
        if snr >= threshold:
            return n
    return N_HARMONICS   # threshold not reached


# ── Step 4: Process one instrument ───────────────────────────────
def process_instrument(instrument):
    print(f"\n[{instrument.upper()}]")

    # Load CTFS data from Module 2
    ctfs_path = CTFS_DIR / f"{instrument}_ctfs.json"
    if not ctfs_path.exists():
        print(f"  NOT FOUND: {ctfs_path}")
        return None

    with open(ctfs_path) as f:
        data = json.load(f)

    frames = data
    print(f"  Loaded {len(frames)} frames")

    # Compute SNR curve averaged over all frames
    snr_curves = []

    for frame_data in frames:
        mags   = np.array(frame_data["magnitudes"])
        phases = np.array(frame_data["phases"])
        f0     = frame_data["f0_hz"]

        # Build original signal from ALL harmonics
        original = reconstruct_signal(
            mags, phases, f0, N_HARMONICS, TARGET_SR
        )

        # Compute SNR for N=1,2,...,32
        snr_per_n = []
        for N in range(1, N_HARMONICS+1):
            reconstructed = reconstruct_signal(
                mags, phases, f0, N, TARGET_SR
            )
            snr = compute_snr(original, reconstructed)
            snr_per_n.append(snr)

        snr_curves.append(snr_per_n)

    # Average SNR curve across all frames
    snr_curves   = np.array(snr_curves)
    mean_snr     = np.mean(snr_curves, axis=0)
    std_snr      = np.std(snr_curves, axis=0)

    # Find N*
    N_star = find_N_star(mean_snr, SNR_THRESHOLD)

    print(f"  Mean SNR at N=32 : {mean_snr[-1]:.1f} dB")
    print(f"  N* (SNR≥{SNR_THRESHOLD}dB) : {N_star}")

    return {
        "instrument" : instrument,
        "mean_snr"   : mean_snr.tolist(),
        "std_snr"    : std_snr.tolist(),
        "N_star"     : N_star,
        "n_frames"   : len(frames)
    }


# ── Step 5: Plot SNR curves ───────────────────────────────────────
def plot_snr_curves(results):
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    colors = {"sitar": "#B85042", "veena": "#A7BEAE", "guitar": "#E7E8D1"}
    bg     = "#1a1a2e"

    fig, ax = plt.subplots(figsize=(10, 6))
    fig.patch.set_facecolor(bg)
    ax.set_facecolor(bg)

    n_values = np.arange(1, N_HARMONICS+1)

    for res in results:
        inst     = res["instrument"]
        mean_snr = np.array(res["mean_snr"])
        std_snr  = np.array(res["std_snr"])
        N_star   = res["N_star"]
        color    = colors.get(inst, "white")

        ax.plot(n_values, mean_snr,
                label=f"{inst.capitalize()} (N*={N_star})",
                color=color, linewidth=2.5)

        ax.fill_between(n_values,
                        mean_snr - std_snr,
                        mean_snr + std_snr,
                        alpha=0.15, color=color)

        # Mark N* point
        ax.axvline(x=N_star, color=color, linestyle='--',
                   alpha=0.5, linewidth=1)
        ax.scatter([N_star], [mean_snr[N_star-1]],
                   color=color, s=80, zorder=5)

    # Draw SNR threshold line
    ax.axhline(y=SNR_THRESHOLD, color='white', linestyle=':',
               linewidth=1.5, label=f'{SNR_THRESHOLD} dB threshold')

    ax.set_xlabel("Number of Harmonics (N)", color='white', fontsize=12)
    ax.set_ylabel("SNR (dB)", color='white', fontsize=12)
    ax.set_title("Reconstruction Fidelity — SNR vs N Harmonics",
                 color='white', fontsize=14, fontweight='bold')
    ax.legend(facecolor='#2a2a3e', labelcolor='white', fontsize=11)
    ax.tick_params(colors='white')
    ax.spines['bottom'].set_color('white')
    ax.spines['left'].set_color('white')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(True, alpha=0.2, color='white')

    plt.tight_layout()
    out_path = FIGURES_DIR / "snr_curves.png"
    plt.savefig(str(out_path), dpi=150, bbox_inches='tight',
                facecolor=bg)
    plt.close()
    print(f"\n  Plot saved: {out_path}")


# ── Step 6: Save N* table ─────────────────────────────────────────
def save_N_star_table(results):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    lines = ["instrument,N_star,snr_at_Nstar,n_frames"]
    for res in results:
        inst   = res["instrument"]
        N_star = res["N_star"]
        snr    = res["mean_snr"][N_star-1]
        lines.append(f"{inst},{N_star},{snr:.2f},{res['n_frames']}")

    out_path = RESULTS_DIR / "N_star_table.csv"
    with open(out_path, "w") as f:
        f.write("\n".join(lines))

    print(f"\n  N* table saved: {out_path}")
    print("\n  ╔══════════════════════════════╗")
    print(  "  ║     N* THRESHOLD TABLE       ║")
    print(  "  ╠══════════════════════════════╣")
    for res in results:
        print(f"  ║  {res['instrument']:<8} │ N* = {res['N_star']:>2}           ║")
    print(  "  ╚══════════════════════════════╝")


# ── Main ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("MODULE 3 — RECONSTRUCTION FIDELITY ANALYSIS")
    print("="*45)

    all_results = []
    for inst in INSTRUMENTS:
        res = process_instrument(inst)
        if res:
            all_results.append(res)

    if all_results:
        plot_snr_curves(all_results)
        save_N_star_table(all_results)

    print("\n" + "="*45)
    print("MODULE 3 COMPLETE")
    print("Check results/figures/snr_curves.png")
    print("Proceed to Module 4 — Feature Vectors")
    print("="*45)