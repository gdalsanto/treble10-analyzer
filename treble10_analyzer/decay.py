from pathlib import Path

import multislope
import numpy as np
import onnxruntime

from .rir_io import load_rir_file
from .sh_sectors import build_sector_matrix, resolve_sh_order

DOUBLE_SLOPE_TIME_RATIO = 1.5
onnxruntime.set_default_logger_severity(4)


def broadband_edc(x):
    """Backward (Schroeder) energy integral, discarding the last 0.5% of samples."""
    x = multislope.discard_last_n_percent(x, 0.5)
    return np.cumsum(x[..., ::-1] ** 2, axis=-1)[..., ::-1]


def estimate_sorted_decay(net, edc):
    """Run multislope on a precomputed broadband EDC, sorted fast -> slow slope."""
    fit = net.estimate(edc, input_is_edc=True)
    order = np.argsort(fit.t, axis=-1)
    t = np.take_along_axis(fit.t, order, axis=-1)
    a = np.take_along_axis(fit.a, order, axis=-1)
    return t, a


def is_double_slope(record):
    t_fast, t_slow = record["t_fast"], record["t_slow"]
    if t_fast <= 0 or t_slow <= 0:
        return False
    return t_slow / t_fast >= DOUBLE_SLOPE_TIME_RATIO


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
    information criterion.
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
