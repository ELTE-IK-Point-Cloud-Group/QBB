"""
Created by Márton Ambrus-Dobai;
"""

import numpy as np
from math import ceil, floor
from sys import stdout

class B_binarization:
    def __init__(self, Er : float = 0.9, m : int = 4):
        self.Er = Er
        self.m = m

    def binarize(self, feature_descriptors : np.ndarray) -> np.ndarray:
        if feature_descriptors.shape[1] % self.m != 0:
            raise ValueError("""Invalid input: the chunk size (m) must be a divisor of
                                the number of columns of the feature_descriptors""")

        print("Py :: Binarize feature descriptor")
        if feature_descriptors.shape[1] % self.m == 0:
            bin = np.zeros(shape=feature_descriptors.shape)

            i = 0
            for point in feature_descriptors:
                # Print out progress
                progress = ceil((i/feature_descriptors.shape[0])*100)
                stdout.write('\r\tBinarizing feature: [{0}] {1}%'.format('#'*(floor(progress/5))+' '*(20-floor(progress/5)), progress))

                # Binarizing units of the point feature descriptor
                bin_point = np.zeros(shape=0)                
                for unit in np.reshape(point, (int(len(point)/self.m), self.m)):
                    bin_point = np.concatenate((bin_point, self._binarize_unit(unit)))
                
                bin[i] = bin_point
                i = i + 1
            
            # Print the final progress
            progress = ceil((i/feature_descriptors.shape[0])*100)
            stdout.write('\r\tBinarizing feature: [{0}] {1}%\n\n\r'.format('#'*(ceil(progress/5))+' '*(20-ceil(progress/5)), progress))

            return bin

    def _binarize_unit(self, S_values : np.ndarray) -> np.ndarray:
        bits = np.zeros(shape=len(S_values))
        vector_pair = self._make_index_value_pairs(S_values)
        S_sum = np.sum(S_values)
        for i in range(self.m):
            temp_sum = sum(v for _,v in vector_pair[:i+1])
            if temp_sum > self.Er * S_sum:
                for k in range(i+1):
                    bits[vector_pair[k][0]] = 1
                return bits
        return bits

    def _make_index_value_pairs(self, values : np.ndarray) -> list:
        ret = [ (i, values[i]) for i in range(len(values)) ]
        ret.sort(key=lambda x:x[1], reverse=True)
        return ret
