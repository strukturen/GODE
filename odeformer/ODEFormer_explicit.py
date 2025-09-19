import numpy as np
import matplotlib.pyplot as plt
import argparse
import torch
import csv
import os
import scipy
from data.utils import *

from odeformer.model import SymbolicTransformerRegressor
from odeformer.metrics import r2_score
from parsers import get_parser

parser = get_parser()
params = parser.parse_args()

model_args = {'beam_size': 20,
              'beam_temperature': 1}

dstr = SymbolicTransformerRegressor(
                    from_pretrained=True, params=params)
dstr.set_model_args(model_args)

# adapt model in parsers: enc_emb_dim and dec_emb_dim
# change model in sklearn_wrapper
DATA_TYPE = 'data_noisy'
noise_sigma = 0.05
example_directory = '../GVAE/examples/'
exec(open("../GVAE/scipy_solver.py").read())
def main(times, trajectory):
    all = dstr.fit(times, trajectory)
    best_eq = all[0][0]
    pred_traj = dstr.predict(times, trajectory[0])
    r2 = r2_score(trajectory, pred_traj)
    print(best_eq)
    return best_eq, r2

if __name__ == '__main__':
    for case in range(0,10):
        # Choose example
        exec(open(example_directory + "odeformer" + str(case) + ".py").read())
        exec(open(example_directory + "odeformer_basic.py").read())
        print('True equation:', true_eq)
        times = np.float64(t_val.squeeze(1).numpy())
        trajectory = np.float64(u_val.numpy())
        #filename = 'Output_explicit_odeformerORG_RS_ID' + str(case) + '.csv'
        filename = 'Output_explicit_odeformer128_RS_ID' + str(case) + '.csv'
        if filename in os.listdir():
            with open(filename, mode = 'a', newline ='') as csvfile:
                csvwriter = csv.writer(csvfile)
                pred_eq, r2 = main(times, trajectory)
                csvwriter.writerow([pred_eq, r2])
        else:
            with open(filename, 'w', newline='') as csvfile:
                csvwriter = csv.writer(csvfile)
                csvwriter.writerow(['Predicted solution', 'R2'])
                pred_eq, r2 = main(times, trajectory)
                csvwriter.writerow([pred_eq, r2])
