# treble10-analyzer

## Content 

This repository contains tools to download, convert, and analyze the Treble10-RIR dataset.
`rir_hoa.py` streams the dataset from Huggingface and saves the RIRs, mic/source positions,
and transfer functions as per-room/source `.npz` files. `analysis.py` provides functions for
common room-acoustic analysis (energy decay curves, RT60, clarity/definition parameters, echo
density) and `edc_explorer.ipynb` is a notebook for interactively visualizing the energy decay
curve per frequency band for a chosen `.npz` file.


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