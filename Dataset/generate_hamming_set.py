"""
This file generates a set of most distant permutations from each other. This is only a support dataset: this will be useful
when we will crop each image and reorder them in those distances.
This is a kind of mathematical distribution, no data are loaded here.
"""

import numpy as np
import itertools
import os
from scipy.spatial.distance import cdist
import h5py
from config import conf

NUM_CROPS = conf.numCrops
NUM_PERMUTATIONS = conf.hammingSetSize
SELECTION = conf.selectionMethod
FILENAME = conf.hammingFileName
FOLDER = conf.resources


def hamming_set(num_crops=NUM_CROPS, num_permutations=NUM_PERMUTATIONS, selection=SELECTION, filename=FILENAME):
    """
    generate and save the hamming set
    :param num_crops: number of tiles from each image
    :param num_permutations: Number of permutations to select (i.e. number of classes for the pretext task)
    :param selection: Sample selected per iteration based on hamming distance: [max] highest; [mean] average
    :param filename: name of file
    :return a list of different permutations: [[perm1], [perm2], ...]. Each permutation is in form (10_elements)
    """
    # create different permutation for num_crops (i.e: num_crops=9, P_hat[0]=(0, 1, 2, 4, 5, 3, 7, 8, 6, 9)
    P_hat = np.array(list(itertools.permutations(list(range(num_crops)), num_crops)))
    n = P_hat.shape[0]  # number of total permutations (i.e num_crops=9 -> shape[0]=3628800

    j = np.random.randint(n)
    P = np.array(P_hat[j]).reshape((1, -1))  # reshape j array into [[1, 2, ...]]

    for _ in range(num_permutations)[1:]:
        # select the <num_permutations> max distant from each other of permutations
        P = np.concatenate([P, P_hat[j].reshape([1, -1])], axis=0)  # concatenate as [[el1], [el2], [...]]
        P_hat = np.delete(P_hat, j, axis=0)
        # Takes the distance between the combination that are already present in P and those who are in P_hat.
        # Note that in P_hat there are no combinations of P.
        D = cdist(P, P_hat, metric='hamming').mean(axis=0).flatten()

        if selection == 'max':
            # select max distances between
            j = D.argmax()
        elif selection == 'mean':
            m = int(D.shape[0] / 2)
            S = D.argsort()
            j = S[np.random.randint(m - 10, m + 10)]

    os.makedirs(FOLDER, exist_ok=True)

    with h5py.File(os.path.join(os.getcwd(), FOLDER, filename + str(NUM_PERMUTATIONS) + '.h5'), 'w') as h5f:
        h5f.create_dataset('max_hamming_set', data=P)

    print('file created --> ' + FOLDER + filename + str(NUM_PERMUTATIONS) + '.h5')


if __name__ == "__main__":
    hamming_set()

