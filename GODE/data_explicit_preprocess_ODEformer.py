# Preprocess data for ODEformer
# ========================
# Karin Yu, 2025

from parser.cfg_parser import *
from tqdm import tqdm
import h5py
import numpy as np
import random
import torch
import sys
from utils import *

data_file = 'data/data_nlode_explicit_odeformer_test.h5'
dump_data_file = '../odeformer/data/data_nlode_explicit_odeformer_test.h5'
grammar_file = 'grammar/nlode_b1.grammar'
dump_data_file_bool = True

# load common arguments
exec(open('grammar/common_args.py').read())

# random seed
seed = 19260817
C_MAX_INT = 5
C_INPUT = 15

random.seed(seed)
np.random.seed(seed)

# load grammar
grammar = Grammar(grammar_file)

# open data
sol_data = []
with h5py.File(data_file, 'r') as f:
    data_eq = f['equations'][:]
    sol_data.extend([f[f'matrices/matrix_{i}'][:] for i in range(len(f['matrices']))])

equations = []
# equations
for i in tqdm(range(len(data_eq))):
    equations.append(str(data_eq[i])[2:-1])

with h5py.File(dump_data_file, 'w') as h5f:
    dt = h5py.string_dtype(encoding='utf-8')
    h5f.create_dataset('equations', data = np.array(equations, dtype = dt))
    group = h5f.create_group('matrices')
    # Iterate through the list and save each matrix as a dataset
    for idx, matrix in enumerate(sol_data):
        # Create a dataset for each matrix
        group.create_dataset(f'matrix_{idx}', data=matrix)

print('Data dumped.')
