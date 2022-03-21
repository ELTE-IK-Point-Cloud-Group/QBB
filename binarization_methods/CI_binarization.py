"""
Created 2021-12-14 by Janos Szalai-Gindl;
"""

import numpy as np

class CI_binarization:
    def __init__(self, encoding_group_length = 11, coding_bits = 2):
        self.encoding_group_length = encoding_group_length
        self.coding_bits = coding_bits
        self.Bcodes = self._get_all_zero_one()
        
    def _get_all_zero_one(self):
        formatation_str = '{0:0' + str(self.coding_bits) + 'b}'
        result_list = []
        for num in range(2**(self.coding_bits)):
            result_list.append([int(x) for x in list(formatation_str.format(num))])
        return np.array(result_list)
    
    def _binarize_unit(self, result, fds_part, start_column):
        fds_abs_diff_from_mean = np.abs(fds_part - np.mean(fds_part, axis=1, keepdims=True))
        fds_std = np.std(fds_part, axis=1, keepdims=True)
        for case_idx in range(1,(2**self.coding_bits) - 1):
            for idx in np.argwhere((fds_abs_diff_from_mean - (case_idx+1)*fds_std < 0) & 
                                   (fds_abs_diff_from_mean - case_idx*fds_std >= 0)):
                result[(idx[0], start_column + idx[1])] = self.Bcodes[case_idx]
        case_idx = (2**self.coding_bits) - 1
        for idx in np.argwhere((fds_abs_diff_from_mean - case_idx*fds_std >= 0)):
            result[(idx[0], start_column + idx[1])] = self.Bcodes[case_idx]
    
    def binarize(self, feature_descriptors):#feature_descriptors.shape = (number of points, dim. of feat. descriptors)
        if feature_descriptors.shape[1] % self.encoding_group_length != 0:
            raise ValueError("""Invalid input: the encoding_group_length must be a divisor of
                                the number of columns of the feature_descriptors""")
        result = np.zeros((feature_descriptors.shape[0], feature_descriptors.shape[1], self.coding_bits), 
                          dtype = np.int8)
        for unit_idx in range(int(feature_descriptors.shape[1]/self.encoding_group_length)):
            start_column = unit_idx * self.encoding_group_length
            end_column = (unit_idx + 1) * self.encoding_group_length
            self._binarize_unit(result, feature_descriptors[:,start_column:end_column], start_column)
        result = result.flatten().reshape((feature_descriptors.shape[0], 
                                           self.coding_bits*feature_descriptors.shape[1]))
        return result