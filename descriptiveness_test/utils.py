"""
Created by Dániel Varga;
"""

import warnings
warnings.filterwarnings("ignore")

from gmpy2 import mpz, hamdist, unpack
import matplotlib.pyplot as plt
import numpy as np
import sys
import json

def to_arr(s : str, l = 6):
    temp = unpack(mpz(s), 1)
    temp2 = [int(x) for x in temp]
    if len(temp2) < l:
        temp2.extend([0 for i in range(l - len(temp2))])
    return np.array(temp2)

def my_nn_custom(query_value, search_space):
    f = open("test_settings.json","r")
    bits = json.load(f)["bits"]
    qv_ne_sspace = np.hsplit((query_value != search_space), np.cumsum(bits)[:-1])
    weighted_hamming_distances = np.sum(np.array([np.count_nonzero(qv_ne_sspace[idx], axis=1)/bits[idx] for idx in range(len(bits))]),axis=0)
    argsorted_whd = np.argsort(weighted_hamming_distances)
    fst_ind = argsorted_whd[0]
    snd_ind = argsorted_whd[1]
    fst_min = weighted_hamming_distances[fst_ind]
    snd_min = weighted_hamming_distances[snd_ind]
    return [[fst_min, snd_min]],[[fst_ind, snd_ind]]

def my_nn(query_value, search_space):
    fst_min = sys.maxsize
    snd_min = sys.maxsize
    fst_ind = -1
    snd_ind = -1
    for i in np.arange(len(search_space)):
        d = hamdist(query_value, search_space[i])

        if d < fst_min:
            snd_min = fst_min
            fst_min = d
            snd_ind = fst_ind
            fst_ind = i
        elif d < snd_min:
            snd_min = d
            snd_ind = i
    return [[fst_min, snd_min]],[[fst_ind, snd_ind]]

def intersect2D(a, b):
  return np.array([x for x in set(tuple(x) for x in a) & set(tuple(x) for x in b)])

def plot_prc(rec, prec, save_name, viz):
    plt.figure(figsize=(15,10))

    plt.plot(rec, prec, 'xb-', linestyle='solid')

    plt.xlabel('Recall')
    plt.ylabel('Precision')

    plt.savefig(save_name)
    if viz:
        plt.show()
