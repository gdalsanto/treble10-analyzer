import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import multislope
import numpy as np
import spaudiopy as spa

MAX_T_DESIGN_DEGREE = 21  # spa.grids.load_t_design only has designs up to this degree
DOUBLE_SLOPE_TIME_RATIO = 1.5


def load_rir_file(path):
    """Load one rir_hoa.py .npz file, reshaped to (n_mics, n_samples, n_hoa)."""
    data = np.load(path)
    fs = int(data["fs"])
    n_mics = data["posMic"].shape[0]
    rir = data["rir"].reshape(n_mics, -1, data["rir"].shape[-1])
    return rir, fs


def broadband_edc(x):
    """Backward (Schroeder) energy integral, discarding the last 0.5% of samples."""
    x = multislope.discard_last_n_percent(x, 0.5)
    return np.cumsum(x[..., ::-1] ** 2, axis=-1)[..., ::-1]


def build_sector_matrix(sh_order):
    """SH -> spatial-sector matrix, following `sector_edc_loss.sh2sec` in loss.py.

    Uses the smallest t-design valid for `sh_order`, i.e. degree 2*sh_order
    (the minimum required for exact SH quadrature over the sector grid).
    """
    t_design_order = 2 * sh_order
    if t_design_order > MAX_T_DESIGN_DEGREE:
        raise ValueError(
            f"sh_order={sh_order} needs a t-design of degree {t_design_order}, but only "
            f"degrees up to {MAX_T_DESIGN_DEGREE} are available (sh_order <= {MAX_T_DESIGN_DEGREE // 2})."
        )
    sec_azi, sec_zen, _ = spa.utils.cart2sph(*spa.grids.load_t_design(t_design_order).T)
    c_n = spa.sph.maxre_modal_weights(sh_order)
    a_sh2sec, _ = spa.sph.design_sph_filterbank(sh_order, sec_azi, sec_zen, c_n, mode="perfect")
    return a_sh2sec  # (n_sectors, n_sh_channels)


def resolve_sh_order(n_hoa, requested_sh_order):
    """The recording's native ambisonics order, and the order to actually use for sectors.

    Passing a smaller `requested_sh_order` truncates the ACN/SH channels before
    the sector beamformer, trading spatial resolution for a coarser (and,
    for the Bayesian analysis, much cheaper) sector decomposition.
    """
    native_sh_order = round(np.sqrt(n_hoa)) - 1
    if requested_sh_order is None:
        return native_sh_order, native_sh_order
    if requested_sh_order < 1:
        raise ValueError(f"sh_order must be at least 1, got {requested_sh_order}.")
    if requested_sh_order > native_sh_order:
        raise ValueError(
            f"Requested sh_order={requested_sh_order} exceeds the recording's native order "
            f"{native_sh_order} ({n_hoa} HOA channels)."
        )
    return native_sh_order, requested_sh_order


def estimate_sorted_decay(net, edc):
    """Run multislope on a precomputed broadband EDC, sorted fast -> slow slope."""
    fit = net.estimate(edc, input_is_edc=True)
    order = np.argsort(fit.t, axis=-1)
    t = np.take_along_axis(fit.t, order, axis=-1)
    a = np.take_along_axis(fit.a, order, axis=-1)
    return t, a


def analyze_directory(data_dir, n_slopes=2, sh_order=None, max_files=None):
    """Run the omni-channel and sector-domain decay analysis over all .npz files.

    `sh_order` optionally truncates the recordings' ACN/SH channels to a lower
    ambisonics order before the sector beamformer (default: use each
    recording's native order).
    """
    npz_files = sorted(Path(data_dir).rglob("*.npz"))
    if max_files is not None:
        npz_files = npz_files[:max_files]
    if not npz_files:
        raise FileNotFoundError(f"No .npz files found under {data_dir}")

    nets_by_fs = {}
    sector_matrix_by_order = {}
    omni_records, sector_records = [], []

    for i_file, path in enumerate(npz_files):
        rir, fs = load_rir_file(path)
        n_mics, _, n_hoa = rir.shape
        room = path.parent.name

        native_sh_order, this_sh_order = resolve_sh_order(n_hoa, sh_order)
        truncated_note = f", using sh_order={this_sh_order} for sectors" if this_sh_order < native_sh_order else ""
        print(f"[{i_file + 1}/{len(npz_files)}] {room}/{path.name}  "
              f"({n_mics} mics, {n_hoa} HOA channels{truncated_note})")

        if fs not in nets_by_fs:
            nets_by_fs[fs] = multislope.DecayFitNet(n_slopes=n_slopes, sample_rate=fs)
        net = nets_by_fs[fs]

        # -- omnidirectional channel (ACN channel 0 = W) --
        omni_edc = broadband_edc(rir[:, :, 0])  # (n_mics, n_samples)
        t, a = estimate_sorted_decay(net, omni_edc)
        for i_mic in range(n_mics):
            omni_records.append({
                "room": room, "file": path.name, "mic": i_mic,
                "t_fast": t[i_mic, 0], "t_slow": t[i_mic, 1],
                "a_fast": a[i_mic, 0], "a_slow": a[i_mic, 1],
            })

        # -- spatial sector domain --
        if this_sh_order not in sector_matrix_by_order:
            sector_matrix_by_order[this_sh_order] = build_sector_matrix(this_sh_order)
        a_sh2sec = sector_matrix_by_order[this_sh_order]
        n_sectors = a_sh2sec.shape[0]
        sector_rir = rir[:, :, :(this_sh_order + 1) ** 2]

        for i_mic in range(n_mics):
            sector_sigs = (sector_rir[i_mic] @ a_sh2sec.T).T  # (n_sectors, n_samples)
            sector_edc = broadband_edc(sector_sigs)
            t, a = estimate_sorted_decay(net, sector_edc)
            for i_sec in range(n_sectors):
                sector_records.append({
                    "room": room, "file": path.name, "mic": i_mic, "sector": i_sec,
                    "t_fast": t[i_sec, 0], "t_slow": t[i_sec, 1],
                    "a_fast": a[i_sec, 0], "a_slow": a[i_sec, 1],
                })

    return omni_records, sector_records


def analyze_directory_bayesian(data_dir, sh_order=None, max_files=None, n_iterations=50, seed=0):
    """BIC-selected (1-3 slope) decay analysis, mirroring `analyze_directory`.

    Unlike `analyze_directory` (which fits a fixed n_slopes with the trained
    DecayFitNet), this uses multislope's BayesianDecayAnalysis with
    n_slopes=0, so the model order is picked per RIR by the Bayesian
    information criterion. Bayesian slice sampling is orders of magnitude
    slower than DecayFitNet, so expect this to dominate runtime once sectors
    are included too. `sh_order` optionally truncates the sector beamformer
    to a lower ambisonics order (see `analyze_directory`), which also cuts
    down that cost since fewer sectors are needed at a lower order.
    """
    npz_files = sorted(Path(data_dir).rglob("*.npz"))
    if max_files is not None:
        npz_files = npz_files[:max_files]
    if not npz_files:
        raise FileNotFoundError(f"No .npz files found under {data_dir}")

    analyzers_by_fs = {}
    sector_matrix_by_order = {}
    omni_records, sector_records = [], []

    for i_file, path in enumerate(npz_files):
        rir, fs = load_rir_file(path)
        n_mics, _, n_hoa = rir.shape
        room = path.parent.name

        native_sh_order, this_sh_order = resolve_sh_order(n_hoa, sh_order)
        truncated_note = f", using sh_order={this_sh_order} for sectors" if this_sh_order < native_sh_order else ""
        print(f"[Bayesian {i_file + 1}/{len(npz_files)}] {room}/{path.name}  "
              f"({n_mics} mics, {n_hoa} HOA channels{truncated_note})")

        if fs not in analyzers_by_fs:
            analyzers_by_fs[fs] = multislope.BayesianDecayAnalysis(
                n_slopes=0, sample_rate=fs, n_iterations=n_iterations, seed=seed,
            )
        analyzer = analyzers_by_fs[fs]

        # -- omnidirectional channel (ACN channel 0 = W) --
        omni_edc = broadband_edc(rir[:, :, 0])  # (n_mics, n_samples)
        fit = analyzer.estimate(omni_edc, input_is_edc=True)
        n_slopes = fit.n_slopes  # (n_mics,), 1-3, selected by BIC
        for i_mic in range(n_mics):
            omni_records.append({
                "room": room, "file": path.name, "mic": i_mic,
                "n_slopes": int(n_slopes[i_mic]),
                "t_1": fit.t[i_mic, 0], "t_2": fit.t[i_mic, 1], "t_3": fit.t[i_mic, 2],
                "a_1": fit.a[i_mic, 0], "a_2": fit.a[i_mic, 1], "a_3": fit.a[i_mic, 2],
                "noise": fit.n[i_mic, 0],
            })

        # -- spatial sector domain --
        if this_sh_order not in sector_matrix_by_order:
            sector_matrix_by_order[this_sh_order] = build_sector_matrix(this_sh_order)
        a_sh2sec = sector_matrix_by_order[this_sh_order]
        n_sectors = a_sh2sec.shape[0]
        sector_rir = rir[:, :, :(this_sh_order + 1) ** 2]

        for i_mic in range(n_mics):
            sector_sigs = (sector_rir[i_mic] @ a_sh2sec.T).T  # (n_sectors, n_samples)
            sector_edc = broadband_edc(sector_sigs)
            fit = analyzer.estimate(sector_edc, input_is_edc=True)
            n_slopes = fit.n_slopes  # (n_sectors,), 1-3, selected by BIC
            for i_sec in range(n_sectors):
                sector_records.append({
                    "room": room, "file": path.name, "mic": i_mic, "sector": i_sec,
                    "n_slopes": int(n_slopes[i_sec]),
                    "t_1": fit.t[i_sec, 0], "t_2": fit.t[i_sec, 1], "t_3": fit.t[i_sec, 2],
                    "a_1": fit.a[i_sec, 0], "a_2": fit.a[i_sec, 1], "a_3": fit.a[i_sec, 2],
                    "noise": fit.n[i_sec, 0],
                })

    return omni_records, sector_records


def is_double_slope(record):
    t_fast, t_slow = record["t_fast"], record["t_slow"]
    if t_fast <= 0 or t_slow <= 0:
        return False
    return t_slow / t_fast >= DOUBLE_SLOPE_TIME_RATIO


def summarize_by_room(records, label):
    rooms = sorted(set(r["room"] for r in records))
    print(f"\n{label}: double-slope fraction per room (T_slow/T_fast >= {DOUBLE_SLOPE_TIME_RATIO})")
    for room in rooms:
        room_records = [r for r in records if r["room"] == room]
        n_double = sum(is_double_slope(r) for r in room_records)
        print(f"  {room:30s} {n_double:5d} / {len(room_records):5d}  ({100 * n_double / len(room_records):5.1f}%)")


def write_csv(records, path):
    if not records:
        return
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0].keys()))
        writer.writeheader()
        writer.writerows(records)


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


def process_directory(data_dir, output_dir, n_slopes, sh_order, max_files,
                       skip_bayesian, bayesian_n_iterations, bayesian_seed):
    """Run the full analysis over one directory of .npz files and write its results to output_dir."""
    output_dir.mkdir(parents=True, exist_ok=True)

    omni_records, sector_records = analyze_directory(
        data_dir, n_slopes=n_slopes, sh_order=sh_order, max_files=max_files,
    )

    write_csv(omni_records, output_dir / "omni_t60.csv")
    write_csv(sector_records, output_dir / "sector_t60.csv")

    summarize_by_room(omni_records, "Omnidirectional channel")
    summarize_by_room(sector_records, "Spatial sectors")

    room_summary = summarize_t60_by_room(omni_records, "omni") + summarize_t60_by_room(sector_records, "sector")
    write_csv(room_summary, output_dir / "t60_summary_by_room.csv")

    plot_t60_histograms(omni_records, sector_records, output_dir)

    if not skip_bayesian:
        bayesian_omni_records, bayesian_sector_records = analyze_directory_bayesian(
            data_dir, sh_order=sh_order, max_files=max_files,
            n_iterations=bayesian_n_iterations, seed=bayesian_seed,
        )
        write_csv(bayesian_omni_records, output_dir / "bayesian_omni_t60.csv")
        write_csv(bayesian_sector_records, output_dir / "bayesian_sector_t60.csv")

        bayesian_summary = (
            summarize_bayesian_slope_counts(bayesian_omni_records, "omni", "omnidirectional channel")
            + summarize_bayesian_slope_counts(bayesian_sector_records, "sector", "spatial sectors")
        )
        write_csv(bayesian_summary, output_dir / "bayesian_slope_counts_by_room.csv")

    print(f"\nSaved CSVs and histogram plot to {output_dir}")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_dir", type=str, required=True,
                         help="Directory containing subfolders to process, each searched recursively "
                              "for .npz files from rir_hoa.py")
    parser.add_argument("--output_dir", type=str, default="./t60_results")
    parser.add_argument("--n_slopes", type=int, default=2)
    parser.add_argument("--sh_order", type=int, default=None,
                         help="Truncate to this ambisonics order before the sector beamformer "
                              "(default: use each recording's native order). The sector beamformer "
                              "always uses the smallest valid t-design for the order in use, i.e. "
                              "degree 2*sh_order.")
    parser.add_argument("--max_files", type=int, default=None, help="Only process the first N files per subfolder (for a quick test run)")
    parser.add_argument("--skip_bayesian", action="store_true",
                         help="Skip the Bayesian (BIC) slope-count analysis, which is much slower "
                              "than the fixed-n_slopes DecayFitNet analysis")
    parser.add_argument("--bayesian_n_iterations", type=int, default=50,
                         help="Slice-sampling iterations per model order for the Bayesian (BIC) analysis")
    parser.add_argument("--bayesian_seed", type=int, default=0, help="Seed for the Bayesian slice sampler")
    return parser.parse_args()


def main(args):
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)

    subdirs = sorted(p for p in data_dir.iterdir() if p.is_dir())
    if not subdirs:
        raise FileNotFoundError(f"No subfolders found under {data_dir}")

    for i_dir, subdir in enumerate(subdirs):
        print(f"\n=== [{i_dir + 1}/{len(subdirs)}] {subdir.name} ===")
        process_directory(
            subdir, output_dir / subdir.name,
            n_slopes=args.n_slopes, sh_order=args.sh_order, max_files=args.max_files,
            skip_bayesian=args.skip_bayesian,
            bayesian_n_iterations=args.bayesian_n_iterations, bayesian_seed=args.bayesian_seed,
        )


if __name__ == "__main__":
    main(parse_args())
