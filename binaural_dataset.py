#
# Data generator for training the SELDnet
#

import numpy as np
import os
import pandas as pd
import random
import soundfile as sf


class BinauralDataLoader:
    def __init__(
            self, params, shuffle=True, split="train", mode="Noisy"
    ):
        self._batch_size = params["training"]["batch_size"]
        dataset_path = params["path_binaural"]
        dataset_path_test = params["path_binaural_test"]

        self._shuffle = shuffle
        self._mode = mode

        if split == "train":
            self._dir = os.path.join(dataset_path, f"{mode}_trainset")
            self.labels_path = os.path.join(dataset_path, "metaData_trainset.csv")
        elif split == "validation":
            self._dir = os.path.join(dataset_path, f"{mode}_valset")
            self.labels_path = os.path.join(dataset_path, "metaData_valset.csv")
        elif split == "test":
            self._dir = os.path.join(dataset_path_test, f"{mode}_testset")    
            self.labels_path = os.path.join(dataset_path_test, "metaData_testset.csv")
        else:
            raise ValueError("split must be train, validation or test")

        self._filenames_list = os.listdir(self._dir)
        self._labels = pd.read_csv(self.labels_path, header=None, names=["filename", "label", "elevation"])

        self._nb_total_batches = int(np.floor(len(self) / float(self._batch_size)))
                           
    def get_total_batches_in_data(self):
        return self._nb_total_batches

    def __len__(self):
        return len(self._filenames_list)

    def get_batch(self):
        """
        Generates batches of samples
        :return: 
        """
        if self._shuffle:
            random.shuffle(self._filenames_list)
        
        sig_buffer = []
        label_buffer = []

        for i in range(len(self)):
            filename = self._filenames_list[i]
            signal, sr = sf.read(os.path.join(self._dir, filename))
            # breakpoint()
            azimuth_deg = self._labels[self._labels["filename"] == filename]["label"].values[0]

            azimuth_cart = self.deg_to_cart(azimuth_deg)

            sig_buffer.append(signal)
            label_buffer.append(azimuth_cart)

            if len(sig_buffer) == self._batch_size:
                yield np.array(sig_buffer), np.array(label_buffer)
                sig_buffer = []
                label_buffer = []

    def shuffle(self):
        random.shuffle(self._filenames_list)

    def deg_to_cart(self, azimuth_deg):
        "Convert azimuth in degrees, between [-90, 90] to 3D Cartesian coordinates, at z=0"

        # 1. Convert to radians
        
        azimuth_rad = np.deg2rad(azimuth_deg)
        x = np.cos(azimuth_rad)
        y = np.sin(azimuth_rad)
        z = 0

        return np.array([x, y, z])
