"""
Make data splits for the dataloader.
Author: Bharat
Cite: LoopReg: Self-supervised Learning of Implicit Surface Correspondences, Pose and Shape for 3D Human Mesh Registration, NeurIPS' 20.
"""

import numpy as np
from glob import glob
from tqdm import tqdm
import pickle as pkl
import os
from os.path import split, join, exists


# Set the data path here. To use our dataloaders the directory structure should be as follows:
# -DATA_PATH
# --datasets-1
# ---subject-1
# ---subject-2
# --datasets-2

DATA_PATH = '/BS/bharat-2/static00/learnt_registration'
# add the folders you want to be the part of this dataset. Typically these would be the folders in side DATA_PATH
datasets = ['axyz', 'renderpeople', 'renderpeople_rigged', 'th_good_1', 'th_good_3', 'julian', 'treedy']


def function(datasets, split, save_path):
    lis = []
    for dataset in datasets:
        for scan in tqdm(glob(join(DATA_PATH, dataset, '*'))):
            lis.append(scan)

    print(len(lis))

    tr_lis, te_lis, count = [], [], 0
    for dataset in datasets:
        for scan in tqdm(glob(join(DATA_PATH, dataset, '*'))):
            if count > split * len(lis):
                tr_lis.append(scan)
            else:
                te_lis.append(scan)
            count += 1

    print('train', len(tr_lis), 'test', len(te_lis))
    pkl.dump({'train': tr_lis, 'val': te_lis},
             open(save_path, 'wb'))


def main():
    SAVE_PATH = 'assets/data_split_01.pkl'
    SPLIT = 0.1

    function(datasets, SPLIT, SAVE_PATH)


if __name__ == "__main__":
    main()


