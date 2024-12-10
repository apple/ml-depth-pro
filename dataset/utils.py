import h5py
import numpy as np


def get_hdf5_array(file_path):
    keys = []
    with h5py.File(file_path, 'r') as file:
        return file['dataset'][:].astype(np.float32)
