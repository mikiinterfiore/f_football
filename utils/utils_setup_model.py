import os
import pickle
import json
import pandas as pd
import numpy as np

_BASE_DIR = '/home/costam/Documents'
_DATA_DIR = os.path.join(_BASE_DIR, 'fantacalcio/data')


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


def assign_label_relative_importance(y_train, y_test, label_encoder):

    # relative_imp = [30, 15, 5, 1, 5, 15, 30] # changed on 23/11/2020
    relative_imp = [5, 4, 3, 1, 3, 4, 5]

    y_tot = np.concatenate((y_train, y_test))
    y_labels_freq = np.unique(y_tot, return_counts=True)
    # original_labels_freq = dict(zip(label_encoder.classes_,
    #                                 np.max(y_labels_freq[1])/y_labels_freq[1]))
    labels_weight_map = dict(zip(label_encoder.classes_,  relative_imp))
    encoded_weight_map = dict(zip(y_labels_freq[0], relative_imp))
    y_tot = pd.DataFrame({'label' : y_tot})
    y_tot['weight'] = y_tot['label'].map(encoded_weight_map)
    scale_weight = y_tot['weight'].copy()

    encoded_map = dict(zip(map(str, label_encoder.classes_),
                           map(int, np.unique(y_tot))))
    encoded_map_filename = 'xgboost_softmax_label_encoder.pkl'
    encoded_map_filename = os.path.join(_DATA_DIR, 'models', encoded_map_filename)
    # Open the file to save as pkl files
    with open(encoded_map_filename, 'w') as f:
        json.dump(encoded_map, f)

    return scale_weight, labels_weight_map
