"""
Created by Márton Ambrus-Dobai;
"""

import numpy as np

class Dim_Binarizer:
    bin_arrays = {}

    def __init__(self) -> None:
        self.bin_arrays = {}

    def binarize(self, data : np.ndarray, endpoints : list) -> np.ndarray:
        result : np.ndarray
        endpoints = endpoints[1:-1] if 2 < len(endpoints) else endpoints
        groups = len(endpoints)+1
        bin_arr = self.get_bin_arrays(groups)
        places = np.searchsorted(endpoints, data, side="left")
        f = lambda x : bin_arr[x]
        result = f(places)
        return result
    
    def get_bin_arrays(self, groups : int) -> np.ndarray:
        if groups not in self.bin_arrays:
            self.generate_bin_arrays(groups)
        return self.bin_arrays[groups]

    def generate_bin_arrays(self, groups : int) -> None:
        bin_array : np.ndarray
        bits = int(np.ceil(np.log2(groups)))
        format_str = '{0:0'+str(bits)+'b}'
        for i in range(2**bits):
            list = [int(x) for x in format_str.format(i)]
            arr = np.array([list])
            if 0 == i:
                bin_array = arr
            else:
                bin_array = np.concatenate((bin_array, arr))
        self.bin_arrays[groups] = bin_array
