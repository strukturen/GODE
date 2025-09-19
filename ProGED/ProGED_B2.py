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
exec(open("../GVAE/scipy_solver.py").read())

NN_LOSS_DEFINED = False
NN_3 = False # Use three NN instead of one

if DATA_TYPE == 'data':
    noise_sigma = 0
elif DATA_TYPE == 'data_noisy':
    print('Noisy case')
    noise_sigma = 0.05

def main(t_val, u_val, du_dt_NN, d2u_dt2_NN=None):
    if d2u_dt2_NN is None:
        data = pd.DataFrame({'t': t_val, 'x0': u_val, 'x1': du_dt_NN})
        model = EqDisco(
            data=data,
            task_type='differential',
            lhs_vars=['x1', 'x0'],
            rhs_vars=['t', 'x0'],
            system_size=2,
            strategy_settings={"max_repeat": 100},
            sample_size = 100,
            generator_template_name="universal",
            verbosity = 0,
            estimation_settings = {"max_constants": 10},
        )
    else:
        data = pd.DataFrame({'t': t_val, 'x0': u_val, 'x1': du_dt_NN, 'x2': d2u_dt2_NN})
        model = EqDisco(
            data=data,
            task_type='differential',
            lhs_vars=['x2', 'x1', 'x0'],
            rhs_vars=['t', 'x0', 'x1'],
            system_size=3,
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
    for i in range(0,1): # Choose number
        case = str(i)
        # benchmark 2
        #exec(open("examples/ode" + case + ".py").read())
        # t_val = np.linspace(t_low, t_high + 1 / fs, int((t_high - t_low) * fs + 1))
        # benchmark 3
        exec(open("../GVAE/examples/pendulum.py").read())
        exec(open("../GVAE/examples/de_base_data.py").read())
        #filename = 'Output_B2_ProGED_ODE'+case+'_best.csv'
        filename = 'Output_ProGED_pendulum_best.csv'
        with open(filename, 'w', newline='') as csvfile:
            csvwriter = csv.writer(csvfile)
            csvwriter.writerow(['Predicted solution', 'Loss'])
            for _ in range(5): # iteration
                if ORDER == 2:
                    eq, loss = main(t_val.squeeze(), u_val.squeeze(), du_dt_NN.detach().squeeze(), d2u_dt2_NN.detach().squeeze())
                    #eq, loss = main(t_val.squeeze(), u_val.squeeze(), du_dt_val.squeeze(),d2u_dt2_val.squeeze())
                else:
                    eq, loss = main(t_val.squeeze(), u_val.squeeze(), du_dt_NN.detach().squeeze())
                csvwriter.writerow([eq, loss])