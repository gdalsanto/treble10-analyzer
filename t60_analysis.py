import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import multislope
import numpy as np

import spaudiopy as spa

DEFAULT_T_DESIGN_ORDER = 4
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


def build_sector_matrix(sh_order, t_design_order):
    """SH -> spatial-sector matrix, following `sector_edc_loss.sh2sec` in loss.py."""
    sec_azi, sec_zen, _ = spa.utils.cart2sph(*spa.grids.load_t_design(t_design_order).T)
    c_n = spa.sph.maxre_modal_weights(sh_order)
    a_sh2sec, _ = spa.sph.design_sph_filterbank(sh_order, sec_azi, sec_zen, c_n, mode="perfect")
    return a_sh2sec  # (n_sectors, n_sh_channels)


def estimate_sorted_decay(net, edc):
    """Run multislope on a precomputed broadband EDC, sorted fast -> slow slope."""
    fit = net.estimate(edc, input_is_edc=True)
    order = np.argsort(fit.t, axis=-1)
    t = np.take_along_axis(fit.t, order, axis=-1)
    a = np.take_along_axis(fit.a, order, axis=-1)
    return t, a


def analyze_directory(data_dir, n_slopes=2, t_design_order=DEFAULT_T_DESIGN_ORDER, max_files=None):
    """Run the omni-channel and sector-domain decay analysis over all .npz files."""
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
        print(f"[{i_file + 1}/{len(npz_files)}] {room}/{path.name}  ({n_mics} mics, {n_hoa} HOA channels)")

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
        sh_order = round(np.sqrt(n_hoa)) - 1
        key = (sh_order, t_design_order)
        if key not in sector_matrix_by_order:
            sector_matrix_by_order[key] = build_sector_matrix(sh_order, t_design_order)
        a_sh2sec = sector_matrix_by_order[key]
        n_sectors = a_sh2sec.shape[0]

        for i_mic in range(n_mics):
            sector_sigs = (rir[i_mic] @ a_sh2sec.T).T  # (n_sectors, n_samples)
            sector_edc = broadband_edc(sector_sigs)
            t, a = estimate_sorted_decay(net, sector_edc)
            for i_sec in range(n_sectors):
                sector_records.append({
                    "room": room, "file": path.name, "mic": i_mic, "sector": i_sec,
                    "t_fast": t[i_sec, 0], "t_slow": t[i_sec, 1],
                    "a_fast": a[i_sec, 0], "a_slow": a[i_sec, 1],
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
        ax.set_xlabel("T60 [s]")
        ax.set_title(title)
        ax.legend()
    axes[0].set_ylabel("count")
    fig.suptitle("Two-slope T60 distribution (n_slopes=2)")
    fig.tight_layout()
    fig.savefig(Path(output_dir) / "t60_histograms.png", dpi=150)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    for ax, records, title in zip(axes, [omni_records, sector_records], ["Omnidirectional channel", "Spatial sectors"]):
        rooms = sorted(set(r["room"] for r in records))
        fractions = []
        for room in rooms:
            room_records = [r for r in records if r["room"] == room]
            fractions.append(100 * sum(is_double_slope(r) for r in room_records) / len(room_records))
        ax.barh(rooms, fractions, color="tab:green")
        ax.set_xlabel("Double-slope RIRs [%]")
        ax.set_title(title)
    fig.suptitle("Fraction of RIRs classified as double-slope, per room")
    fig.tight_layout()
    fig.savefig(Path(output_dir) / "double_slope_fraction_by_room.png", dpi=150)
    plt.close(fig)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_dir", type=str, required=True,
                         help="Directory (searched recursively) containing the .npz files from rir_hoa.py")
    parser.add_argument("--output_dir", type=str, default="./t60_results")
    parser.add_argument("--n_slopes", type=int, default=2)
    parser.add_argument("--t_design_order", type=int, default=DEFAULT_T_DESIGN_ORDER)
    parser.add_argument("--max_files", type=int, default=None, help="Only process the first N files (for a quick test run)")
    return parser.parse_args()


def main(args):
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    omni_records, sector_records = analyze_directory(
        args.data_dir, n_slopes=args.n_slopes, t_design_order=args.t_design_order, max_files=args.max_files,
    )

    write_csv(omni_records, output_dir / "omni_t60.csv")
    write_csv(sector_records, output_dir / "sector_t60.csv")

    summarize_by_room(omni_records, "Omnidirectional channel")
    summarize_by_room(sector_records, "Spatial sectors")

    plot_t60_histograms(omni_records, sector_records, output_dir)
    print(f"\nSaved CSVs and histogram plots to {output_dir}")


if __name__ == "__main__":
    main(parse_args())
