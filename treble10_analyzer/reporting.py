import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from .decay import DOUBLE_SLOPE_TIME_RATIO, is_double_slope


def write_csv(records, path):
    if not records:
        return
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0].keys()))
        writer.writeheader()
        writer.writerows(records)


def summarize_by_room(records, label):
    rooms = sorted(set(r["room"] for r in records))
    print(f"\n{label}: double-slope fraction per room (T_slow/T_fast >= {DOUBLE_SLOPE_TIME_RATIO})")
    for room in rooms:
        room_records = [r for r in records if r["room"] == room]
        n_double = sum(is_double_slope(r) for r in room_records)
        print(f"  {room:30s} {n_double:5d} / {len(room_records):5d}  ({100 * n_double / len(room_records):5.1f}%)")


def summarize_t60_by_room(records, domain):
    """Per-room double-slope fraction plus median/std of the fast/slow T60 histograms."""
    rooms = sorted(set(r["room"] for r in records))
    summary = []
    for room in rooms:
        room_records = [r for r in records if r["room"] == room]
        t_fast = np.array([r["t_fast"] for r in room_records])
        t_slow = np.array([r["t_slow"] for r in room_records])
        n_double = sum(is_double_slope(r) for r in room_records)
        summary.append({
            "domain": domain,
            "room": room,
            "n_records": len(room_records),
            "double_slope_fraction": 100 * n_double / len(room_records),
            "t_fast_median": np.median(t_fast),
            "t_fast_std": np.std(t_fast),
            "t_slow_median": np.median(t_slow),
            "t_slow_std": np.std(t_slow),
        })
    return summary


def summarize_bayesian_slope_counts(records, domain, label):
    """Per-room distribution of the BIC-selected number of slopes (1-3)."""
    rooms = sorted(set(r["room"] for r in records))
    print(f"\nBayesian (BIC) slope-count distribution per room, {label}")
    summary = []
    for room in rooms:
        room_records = [r for r in records if r["room"] == room]
        n = len(room_records)
        counts = {k: sum(r["n_slopes"] == k for r in room_records) for k in (1, 2, 3)}
        mean_n_slopes = np.mean([r["n_slopes"] for r in room_records])
        print(f"  {room:30s} "
              f"1-slope {counts[1]:5d} ({100 * counts[1] / n:5.1f}%)  "
              f"2-slope {counts[2]:5d} ({100 * counts[2] / n:5.1f}%)  "
              f"3-slope {counts[3]:5d} ({100 * counts[3] / n:5.1f}%)  "
              f"mean={mean_n_slopes:.2f}")
        summary.append({
            "domain": domain,
            "room": room,
            "n_records": n,
            "frac_1_slope": 100 * counts[1] / n,
            "frac_2_slopes": 100 * counts[2] / n,
            "frac_3_slopes": 100 * counts[3] / n,
            "mean_n_slopes": mean_n_slopes,
        })
    return summary


def plot_t60_histograms(omni_records, sector_records, output_dir, t60_max=5.0):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    bins = np.linspace(0, t60_max, 60)
    for ax, records, title in zip(axes, [omni_records, sector_records], ["Omnidirectional channel", "Spatial sectors"]):
        t_fast = np.clip([r["t_fast"] for r in records], 0, t60_max)
        t_slow = np.clip([r["t_slow"] for r in records], 0, t60_max)
        ax.hist(t_fast, bins=bins, alpha=0.6, label="fast slope", color="tab:blue")
        ax.hist(t_slow, bins=bins, alpha=0.6, label="slow slope", color="tab:orange")
        # set the maximum of the y-axis to the maximum of the two histograms
        ax.set_ylim(0, max(ax.get_ylim()[1], 1)*1.1)
        ax.set_xlabel("T60 [s]")
        ax.set_title(title)
        ax.legend()
    axes[0].set_ylabel("count")
    fig.suptitle("Two-slope T60 distribution (n_slopes=2)")
    fig.tight_layout()
    fig.savefig(Path(output_dir) / "t60_histograms.png", dpi=150)
    plt.close(fig)
