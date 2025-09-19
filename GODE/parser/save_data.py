# file to save parser trees
# ================================
# Karin Yu, 2025

from cfg_parser import *
from tqdm import tqdm
import pickle
import numpy as np
import random
import h5py

#define the following parameters
cfg_grammar_file = '../grammar/nlode_b1.grammar'
data_file = '../data/data_nlode_b1_11k_train.txt'
dump_parse_tree_file = '../data/data_nlode_b1_11k_train.cfg_dump'
dump_parse_trees_bool = False
dump_data_file = '../data/data_nlode_b1_11k_train.h5'
dump_data_file_bool = True
exec(open('../grammar/common_args.py').read())

# load grammar
grammar = Grammar(cfg_grammar_file)

# Check parser for all data
with open(data_file, 'r') as f:
    equations = f.readlines()
for i in range(len(equations)):
    equations[i] = equations[i].strip()

if dump_parse_trees_bool:
    fout_pt = open(dump_parse_tree_file, 'wb')
    for eq in tqdm(equations):
        ts = grammar.parse(eq)
        assert isinstance(ts, list) and len(ts) == 1
        pickle.dump(ts, fout_pt, pickle.HIGHEST_PROTOCOL)
    fout_pt.close()
    print('### All equations are parsed and saved to %s' % dump_parse_tree_file)

if dump_data_file_bool:
    print('### Now generating one-hot encoding for the parsed trees')
    onehot_list = []

    for eq in tqdm(equations):
        onehot_list.append(grammar.create_one_hot(eq, MAX_LEN))
    all_onehot = np.vstack(onehot_list)
    # shuffle training set
    idxes = list(range(all_onehot.shape[0]))
    random.shuffle(idxes)
    idxes = np.array(idxes, dtype=np.int32)
    all_onehot = all_onehot[idxes, :, :]
    print('num samples: ', len(idxes))
    h5f = h5py.File(dump_data_file, 'w')
    h5f.create_dataset('data', data=all_onehot)
    h5f.close()