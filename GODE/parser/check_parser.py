# to demonstrate parser function of chosen grammar
# ================================
# Karin Yu, 2025

from cfg_parser import *
from tqdm import tqdm

# text to modify
cfg_grammar_file = '../grammar/nlode_b1.grammar'
check_data = True
data_file = '../data/data_nlode_b1_11k_train.txt'
# Explicit ODE
eq_check = 'C*u^1+C*u^2+C*t*cos(u*C)'

print('Check equation:', eq_check)
grammar = Grammar(cfg_grammar_file)
ts = grammar.parse(eq_check)
print('Number of parsing trees:', len(ts))
t = ts[0]
print('One hot encoded vector:',  np.argmax(grammar.create_one_hot(eq_check, 100), axis = 2))
# reconstruct based on parsing tree
print('Original equation:', eq_check)
print('Reconstructed equation:', reconstruct(t))

# check masking
print('Check masking')
one_hot = grammar.create_one_hot(eq_check, 100)
reconstructed = grammar.generate_sequence(one_hot)
print('Reconstructed equation from one-hot encoded vector:', reconstructed[0])


if check_data:
    # Check parser for all data
    with open(data_file, 'r') as f:
        equations = f.readlines()
    for i in range(len(equations)):
        equations[i] = equations[i].strip()

    if eq_check in equations:
        print('Contains check equation:', eq_check)
    else:
        print('Does not contain equation.')

    max_len = 0
    max_C_count = 0
    for eq in tqdm(equations):
        ts = grammar.parse(eq)
        max_C_count= max(max_C_count, eq.count('C'))
        assert isinstance(ts, list) and len(ts) == 1
        productions_seq = [tree.productions() for tree in ts]
        rules_seq = [len([grammar._prod_map[prod] for prod in entry]) for entry in productions_seq]
        max_len = max(rules_seq[0], max_len)
    print('Max length:', max_len)
    print('Max number of C:', max_C_count)

