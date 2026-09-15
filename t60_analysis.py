import argparse
from pathlib import Path

from treble10_analyzer.decay import analyze_directory, analyze_directory_bayesian
from treble10_analyzer.reporting import (
    plot_t60_histograms,
    summarize_bayesian_slope_counts,
    summarize_by_room,
    summarize_t60_by_room,
    write_csv,
)


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
