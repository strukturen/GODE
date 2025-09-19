# File to generate parser trees
# ================================
# Karin Yu, 2025

from cfg_parser import *
import numpy as np
import random
import torch

#define the following parameters
cfg_grammar_file = '../grammar/nlode_b1.grammar'
exec(open('../grammar/common_args_explicit.py').read())
data_file_train = '../data/data_nlode_b1_11k_train.txt'
data_file_test = '../data/data_nlode_b1_11k_test.txt'
n_grammar = 11000

# function for preprocessing for CONT5 and CONT6
def preprocess_eq(eq):
    eq_ = eq.split('+')
    for i in range(len(eq_)):
        if eq_[i][0] != 'C':
            eq_[i] = eq_[i][1:]
    eq_list = list(set(eq_))
    eq_list = sorted(eq_list, key=lambda x: len(x))
    return "+".join(eq_list)#+eq_[-1]

# load grammar
grammar = Grammar(cfg_grammar_file, pcfg = False) # turn on or off if PCFG

# add attributes
disallowed_dict = {
}

grammar.add_disallowed(disallowed_dict)

all_data = []
n_rtgae = len(all_data)

# 3. Sobol sampling
soboleng = torch.quasirandom.SobolEngine(dimension=grammar.dim, scramble=True, seed=11)

failed = 0
while len(all_data)-n_rtgae < n_grammar:
    uniform = soboleng.draw(MAX_LEN)
    z = (torch.erfinv(2 * uniform-1) * (2 ** 0.5)) * 2
    z = z.unsqueeze(0)
    try:
        X_hat = grammar.sample_using_masks(z.numpy(), True)  # check here there are many random things
        seq = grammar.generate_sequence(X_hat)
        # pre-process data

        # for benchmark 1
        if X_hat.shape[1] <= MAX_LEN and len(seq[0]) > 0:
            # for benchmark 2 and 3
            # if X_hat.shape[1] <= MAX_LEN and len(seq[0]) > 0 and not seq[0] in all_data and 'diff' in seq[0] and not 'diff(diff(diff(u,t),t),t)' in seq[0]:# and (not seq[0].count('diff') == 1 and '0' in seq[0]):  #exclude order 3 derivatives
            all_data.append(seq[0])
    except:
        failed += 1
    if (len(all_data)-n_rtgae) % 200 == 0:
        print('Generated', len(all_data)-n_rtgae, 'equations')
print('Number of failed generations:', failed)
print('Random shuffle...')

np.random.seed(11)
np.random.shuffle(all_data)

print('Total number of data:', len(all_data))

n_train = n_grammar-1000 # 1,000 test equations
training_data = all_data[:n_train]
test_data = all_data[n_train:]

with open(data_file_train, 'w') as file:
    for string in training_data:
        # Write each string to a new line in the file
        file.write(f"{string}\n")

with open(data_file_test, 'w') as file:
    for string in test_data:
        # Write each string to a new line in the file
        file.write(f"{string}\n")
print('Finished generating and writing')
