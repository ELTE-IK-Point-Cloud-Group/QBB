"""
Created by Márton Ambrus-Dobai;
"""

from typing import Dict
import numpy as np
from sys import stdout
from math import ceil, floor
from QBB_statistics_for_descriptor import Statistics
from QBB_dim_binarizer import Dim_Binarizer

class Binarization():
    def __init__(self, binarizer : Dim_Binarizer):
        self.binarizer = binarizer

    def binarize(self, feature_descriptor : np.ndarray, descriptor_folder_for_endpoints : str) -> np.ndarray:
        stat = Statistics(use_limit=False)
        dim_ends = stat.get_descriptor_statistics(descriptor_folder_for_endpoints)
        return self._binarize(feature_descriptor.T, dim_ends).T

    def _binarize(self, descriptor : np.ndarray, dim_ends : list) -> np.ndarray:
        bin_desc : np.ndarray
        length, count = descriptor.shape
        for i in range(length):
            new_column = self.binarizer.binarize(descriptor[i], dim_ends[i])
            if i == 0:
                bin_desc = new_column.T
            else:
                bin_desc = np.append(bin_desc, new_column.T, 0)

            progress = ceil((i/length)*100)
            stdout.write('\r\tBinarizing feature: [{0}] {1}%'.format('#'*(floor(progress/5))+' '*(20-floor(progress/5)), progress))
        stdout.write('\r\tBinarizing feature: [{0}] {1}%\n\n'.format('#'*20, 100))
        print(bin_desc.shape)
        return bin_desc