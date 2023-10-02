import os
import sys
import numpy as np
import pyDOE
import h5py

from utils import limits, rescale_to_data


def database_design(seed=42, n_sim=1000, criterion='centermaximin'):
    
    print(f'seed = {seed}')
    print(f'n_sim = {n_sim}')
    print(f'criterion = {criterion}')

    np.random.seed(seed)
    unit = pyDOE.lhs(n=len(limits['hyper']), samples=n_sim, criterion=criterion)
    np.random.seed()

    data = {}

    for dim, par in enumerate(limits['hyper']):
        data[par] = rescale_to_data(unit[:, dim], *limits['hyper'][par])
    
    return data
