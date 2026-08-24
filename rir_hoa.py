import argparse
import io
import os
import numpy as np
import soundfile as sf
from pathlib import Path
from datasets import load_dataset, Audio

def str2list (s):
	return [float(x) for x in s.strip('[]').split(',')]


def resolve_dataset_path(path):
    """If path is a huggingface_hub cache-style download (blobs/refs/snapshots),
    resolve it to the actual repo snapshot dir so `datasets` can see the
    README.md split/config metadata. Otherwise return path unchanged."""
    path = Path(path)
    ref_main = path / "refs" / "main"
    if ref_main.is_file():
        commit_hash = ref_main.read_text().strip()
        snapshot_dir = path / "snapshots" / commit_hash
        if snapshot_dir.is_dir():
            return str(snapshot_dir)
    return str(path)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--path_to_dataset", type=str, default="/scratch/elec/t412-asp/Treble10-RIR")
    parser.add_argument("--split", type=str, default="rir_hoa8")
    parser.add_argument("--output_dir", type=str, default="/scratch/elec/t412-asp/treble10-npz")
    return parser.parse_args()


def main(args):
    path_to_dataset = resolve_dataset_path(args.path_to_dataset)
    split = args.split
    output_dir = args.output_dir
    ds = load_dataset(path_to_dataset, streaming=True, split=split, cache_dir="/scratch/work/dalsag1/.cache/huggingface/datasets")
    ds = ds.cast_column("audio", Audio(decode=False))

    # collect all examples, grouped by (room, source) in a single streaming pass
    # for each rir save the rir, the mic position and the source position in a dictionary
    def new_subset():
        return {
            "rir": [],
            "atf": [],
            "atf_mag": [],
            "mic_position": [], 
            "source_position": [],
        }

    def new_extras():
        return {
            "rt60": [],
            "edt": [],
            "c50": [],
            "abs_avr": [],
            "c_freqs": [],
        }

    def save_room(subsets, extras_by_group, last_example_by_group, fs_by_group):
        # save one npz file per (room, source) group
        for key, subset in subsets.items():
            room, source = key
            source_id = int(source[-1])
            last_example = last_example_by_group[key]
            extras = extras_by_group[key]

            subset["source_position"] = str2list(last_example["Source Position"])

            room_description = last_example["Room Description"]
            fs = fs_by_group[key]
            extras["c_freqs"] = str2list(last_example["Frequencies"])

            # save subset as npz file
            dataset_dir = Path(f"{output_dir}/{split}/{room_description}/")
            os.makedirs(dataset_dir, exist_ok=True)
            filename = f"{dataset_dir}/data_s{source_id + 1:04d}.npz"
            np.savez(
                filename,
                rir=np.vstack(subset["rir"]),
                atf=np.vstack(subset["atf"]),
                atf_mag=np.vstack(subset["atf_mag"]),
                posSrc=subset["source_position"],
                posMic=np.vstack(subset["mic_position"]),
                fs=fs,
            )

    subsets = {}
    extras_by_group = {}
    last_example_by_group = {}
    fs_by_group = {}
    current_room = None

    for example in iter(ds):
        room = example["Room"]
        source = example["Source Label"]
        key = (room, source)

        # the stream is grouped by room; once a new room starts, the
        # previous room is complete, so save it and free its memory
        # before accumulating the next one
        if current_room is not None and room != current_room:
            save_room(subsets, extras_by_group, last_example_by_group, fs_by_group)
            subsets = {}
            extras_by_group = {}
            last_example_by_group = {}
            fs_by_group = {}
        current_room = room

        subset = subsets.setdefault(key, new_subset())
        extras = extras_by_group.setdefault(key, new_extras())

        # decode the raw audio bytes ourselves (libsndfile handles arbitrary
        # channel counts, unlike torchcodec/FFmpeg which chokes on the
        # 81-channel HOA8 layout)
        rir, fs = sf.read(io.BytesIO(example["audio"]["bytes"]), always_2d=False)
        fs_by_group[key] = fs
        print(room, source, rir.shape)
        irlen = rir.shape[0]
        # compute the atf and the atf magnitude
        atf = np.fft.rfft(rir, n=irlen, axis=0)
        atf_mag = 20 * np.log10(np.abs(atf))
        # save data
        subset["rir"].append(rir)
        subset["atf"].append(atf)
        subset["atf_mag"].append(atf_mag)
        subset["mic_position"].append(tuple(str2list(example["Receiver Position"])))
        # log last example
        last_example_by_group[key] = example
        # log extras
        extras["rt60"].append(str2list(example["T30"]))
        extras["edt"].append(str2list(example["EDT"]))
        extras["c50"].append(str2list(example["C50"]))
        extras["abs_avr"].append(str2list(example["Average Absorption (Octave Band)"]))

    if current_room is not None:
        save_room(subsets, extras_by_group, last_example_by_group, fs_by_group)

if __name__ == "__main__":
    args = parse_args()
    main(args)