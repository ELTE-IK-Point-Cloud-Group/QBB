"""
Created by Márton Ambrus-Dobai;
"""

import numpy as np
from sys import stdout
from math import floor, ceil

class QBB_MERSENNE_binarization():
    def __init__(self, dimension_endpoints : list):
        self.dim_ends = dimension_endpoints
        self.bin_arrays = {}

    def binarize(self, descriptor : np.ndarray) -> np.ndarray:
        bin_desc : np.ndarray
        descriptor = descriptor.T
        length, count = descriptor.shape
        for i in range(length):
            new_column = self._binarize_unit(descriptor[i], self.dim_ends[i])
            if i == 0:
                bin_desc = new_column.T
            else:
                bin_desc = np.append(bin_desc, new_column.T, 0)

            progress = ceil((i/length)*100)
            stdout.write('\r\tBinarizing feature: [{0}] {1}%'.format('#'*(floor(progress/5))+' '*(20-floor(progress/5)), progress))
        stdout.write('\r\tBinarizing feature: [{0}] {1}%\n\n'.format('#'*20, 100))
        return bin_desc.T

    def _binarize_unit(self, data : np.ndarray, endpoints : list) -> np.ndarray:
        result : np.ndarray
        endpoints = endpoints[1:-1]
        group_num = len(endpoints)+1
        bin_arr = self._get_bin_arrays(group_num)
        places = np.searchsorted(endpoints, data, side="left")
        f = lambda x : bin_arr[x]
        result = f(places)
        return result

    def _get_bin_arrays(self, group_num : int) -> np.ndarray:
        if group_num not in self.bin_arrays:
            self._generate_bin_arrays(group_num)
        return self.bin_arrays[group_num]

    def _generate_bin_arrays(self, group_num : int) -> None:
        bin_array : np.ndarray
        bits = group_num-1
        format_str = '{0:0'+str(bits)+'b}'
        for i in range(group_num):
            val = 2**i-1
            list = [int(x) for x in format_str.format(val)]
            arr = np.array([list], dtype=np.byte)
            if 0 == i:
                bin_array = arr
            else:
                bin_array = np.concatenate((bin_array, arr))
        self.bin_arrays[group_num] = bin_array