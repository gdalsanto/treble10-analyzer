# treble10-analyzer

## Content

This repository contains tools to download, convert, and analyze the Treble10-RIR dataset.

### Structure

```
treble10-analyzer/
├── rir_hoa.py              # step 1: Treble10-RIR (Huggingface) -> per-room/source .npz files
├── t60_analysis.py         # step 2: CLI running the decay analysis and writing CSVs/plots
├── edc_explorer.ipynb      # notebook to inspect the EDC per frequency band of one .npz file
└── treble10_analyzer/      # reusable analysis package
    ├── rir_io.py           # load a rir_hoa.py .npz file as (n_mics, n_samples, n_hoa)
    ├── sh_sectors.py       # ambisonics -> spatial-sector beamforming (t-design), SH-order handling
    ├── decay.py            # multislope decay analysis: fixed n_slopes and Bayesian (BIC) slope count
    ├── reporting.py        # CSV writing, per-room summaries, T60 histograms
    └── analysis.py         # EDC/EDR, RT60, clarity/definition, echo density and mixing time
```

**Pipeline**

1. `rir_hoa.py` streams the dataset from Huggingface and saves the RIRs, mic/source positions and
   transfer functions as `<output_dir>/<split>/<room_description>/data_sXXXX.npz` (one file per room/source).
2. `t60_analysis.py --data_dir <dir>` processes each subfolder of `<dir>` (searched recursively for `.npz` files)
   and, for each one, analyzes the omnidirectional (first ambisonics) channel and the spatial-sector signals
   obtained with `sh_sectors.py`. Decay times are estimated with `multislope` in `decay.py`
   (`--n_slopes` fixed slopes, plus a slower Bayesian slope-count analysis unless `--skip_bayesian` is given).
   `--sh_order` truncates the ambisonics order before beamforming and `--max_files` limits the run for a quick test.
3. `reporting.py` writes the results to `<output_dir>/<subfolder>/`: `omni_t60.csv`, `sector_t60.csv`,
   `t60_summary_by_room.csv`, a T60 histogram plot and, unless skipped, `bayesian_omni_t60.csv`,
   `bayesian_sector_t60.csv` and `bayesian_slope_counts_by_room.csv`.

`analysis.py` is a standalone toolbox of general room-acoustic functions; it is used by `edc_explorer.ipynb`, not by the CLI.


The dataset is on [Huggingface](https://huggingface.co/datasets/treble-technologies/Treble10-RIR). 
This notebook shows you how I would navigate it. 

### Dataset Content 
- contains simulated RIRs in 10 different rooms. Their volume is between $13.83~\text{m}^3$ and $46.08~\text{m}^3$
- each room contains 5 sound sources
- receiver locations are spaced $0.5$~m apart 
- the sampling frequency is of $32$~kHz  

There are three splits in the dataset: 
- "rir_mono" : Single-channel mono RIRs
- "rir_6ch" : 8th-order Ambisonics RIRs (ACN/SN3D format)
- "rir_hoa8" : Six-channel home audio device layout

Each RIR comes with the following metadata 

| Columnn    | Description |
| -------- | ------- |
| audio | Reference to the RIR audio file. |
| Filename  | Filename and relative path of the WAV file.  |
| Room | Short room nickname (e.g., Room1, Room5).    |
| Room Description    | Descriptive room type (e.g., meeting_room, living_room).   |
| Room Volume [m³ | Volume of the room in cubic meters.   |
| Direct Path Length [m]     | Distance between source and receiver.    |
| Source Label / Position  | Label and 3D coordinates of the source.     |
| Receiver Label / Position    | Label and 3D coordinates of the receiver.   |
| Receiver Type  | Receiver configuration (mono, 8th order, or 6-channel).    |
| Frequencies, EDT, T30, C50, Average Absorption     | Octave-band acoustic parameters.  |
| Avg EDT, Avg T30, Avg Absorption  | Broadband summary values.    |

### Downloading the data

You can download the dataset using the [Huggingface CLI](https://huggingface.co/docs/huggingface_hub/en/guides/cli)
```
hf download treble-technologies/Treble10-RIR --repo-type dataset
```
The whole dataset should be ~$12.2$ G

If you want to avoid downloading everything upfront and instead fetch examples as you iterate you can install 
Huggingface's datasets in your environemnt via ```pip install datasets``` and run

```
from dataset import load_dataset
ds = load_dataset("treble-technologies/Treble10-RIR", streaming=True, split="rir_hoa8")
```
More information about the package check Huggingface [website](https://huggingface.co/docs/datasets/en/installation)