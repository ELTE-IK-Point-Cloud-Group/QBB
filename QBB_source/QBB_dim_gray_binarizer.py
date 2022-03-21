"""
Created by Márton Ambrus-Dobai;
"""

import numpy as np
import graycode
from QBB_dim_binarizer import Dim_Binarizer

class Dim_GRAY_Binarizer(Dim_Binarizer):
    def generate_bin_arrays(self, groups : int) -> None:
        bin_array : np.ndarray
        bits = int(np.ceil(np.log2(groups)))
        format_str = '{0:0'+str(bits)+'b}'
        i = 0
        for num in graycode.gen_gray_codes(bits):
            blist = [int(x) for x in list(format_str.format(num))]
            arr = np.array([blist], dtype=np.byte)
            if 0 == i:
                bin_array = arr
            else:
                bin_array = np.concatenate((bin_array, arr))
            i += 1
        self.bin_arrays[groups] = bin_array