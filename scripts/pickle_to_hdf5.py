import sys, os, pickle, h5py, numpy as np
from pycbc.types import FrequencySeries

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# from modules.functions_ver2 import timer_decorator


# @timer_decorator
def convert_pickle_to_hdf5(pickle_filepath, hdf5_filepath):
    with open(pickle_filepath, "rb") as f:
        data = pickle.load(f)

    with h5py.File(hdf5_filepath, "w") as hdf5_file:
        for key, value in data.items():
            key = str(key)
            value = np.array(value)
            hdf5_file.create_dataset(key, data=value)


if __name__ == "__main__":
    pickle_filepath = "data/sys3_template_bank_mcz40.pkl"
    hdf5_filepath = "data/sys3_template_bank_mcz40.hdf5"
    convert_pickle_to_hdf5(pickle_filepath, hdf5_filepath)
