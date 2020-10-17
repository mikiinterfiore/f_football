import os
import pickle
import json
import pandas as pd
import numpy as np


def bin_target_values(v):

    neg_fv_groups = [-20] + np.arange(-2.5, 0, 2).tolist()
    pos_fv_groups = [0] + np.arange(0.5, 5.5, 2).tolist() + [20]
    fv_groups = neg_fv_groups + pos_fv_groups
    bins = []
    labels = []
    for i in range(1,len(fv_groups)):
        bins.append((fv_groups[i-1], fv_groups[i]))
        labels.append(fv_groups[i-1])
    bins = pd.IntervalIndex.from_tuples(bins, closed='left') # in this way 0 is counted as positive

    labels_dict = dict(zip(bins, labels))
    v_bin = pd.cut(v, bins, include_lowest=True)
    v_bin = v_bin.map(labels_dict)

    return v_bin
