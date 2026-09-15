import numpy as np


def load_rir_file(path):
    """Load one rir_hoa.py .npz file, reshaped to (n_mics, n_samples, n_hoa)."""
    data = np.load(path)
    fs = int(data["fs"])
    n_mics = data["posMic"].shape[0]
    rir = data["rir"].reshape(n_mics, -1, data["rir"].shape[-1])
    return rir, fs
