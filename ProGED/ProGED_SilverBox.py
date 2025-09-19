import sys
import symengine as spe
import torch
from utils import *
import csv
from ProGED import EqDisco
import pandas as pd
import numpy as np

random_seed = 0
torch.manual_seed(random_seed)
np.random.seed(random_seed)

DATA_TYPE = 'data_noisy'
t = spe.symbols('t')
exec(open("../GODE/scipy_solver.py").read())

NN_LOSS_DEFINED = False
NN_3 = False # Use three NN instead of one

if DATA_TYPE == 'data':
    noise_sigma = 0
elif DATA_TYPE == 'data_noisy':
    print('Noisy case')
    noise_sigma = 0.05

def main(t_val, u_val, du_dt_val, d2u_dt2_val, f_val):
    data = pd.DataFrame({'t': t_val, 'x0': u_val, 'x1': du_dt_val, 'x2': d2u_dt2_val, 'x3': f_val})
    model = EqDisco(
        data=data,
        task_type='differential',
        lhs_vars=['x3', 'x2', 'x1', 'x0'], # Equation = F
        rhs_vars=['t', 'x0', 'x1', 'x2'],
        system_size=4,
        strategy_settings={"max_repeat": 200},
        sample_size=100,
        generator_template_name="universal",
        verbosity=0,
        estimation_settings={"max_constants": 50},
    )
    model.generate_models() # Generating models
    model.fit_models() # Estimating parameters
    res = model.get_results()
    candidates, errors = [], []
    for eq in res:
        candidates.append(eq)
        errors.append(eq.get_error())
    order = np.argsort(errors)
    return candidates[order[0]], errors[order[0]]

if __name__ == '__main__':
    # Silver box dataset
    exec(open("../GODE/examples/silver_box.py").read())
    filename = 'Output_ProGED_silver_box_best.csv'
    with open(filename, 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(['Predicted solution', 'Loss'])
        for _ in range(5): # iteration
            eq, loss = main(t_val, u_val, du_dt_val, d2u_dt2_val, f_val.detach().numpy())
            csvwriter.writerow([eq, loss])