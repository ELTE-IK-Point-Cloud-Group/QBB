"""
Created by Márton Ambrus-Dobai;
"""

import numpy as np
from QBB_dim_binarizer import Dim_Binarizer

class Dim_MERSENNE_Binarizer(Dim_Binarizer):
    def generate_bin_arrays(self, groups : int) -> None:
        bin_array : np.ndarray
        bits = groups-1
        format_str = '{0:0'+str(bits)+'b}'
        for i in range(groups):
            val = 2**i-1
            list = [int(x) for x in format_str.format(val)]
            arr = np.array([list], dtype=np.byte)
            if 0 == i:
                bin_array = arr
            else:
                bin_array = np.concatenate((bin_array, arr))
        self.bin_arrays[groups] = bin_array