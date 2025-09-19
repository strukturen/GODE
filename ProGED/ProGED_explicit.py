import sys
import symengine as spe
import torch
from utils import *
import csv
from ProGED import EqDisco
import pandas as pd

random_seed = 11
torch.manual_seed(random_seed)

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

def main(t_val, u_val, du_dt_NN):
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
    )
    model.generate_models() # Generating models
    model.fit_models() # Estimating parameters
    res = model.get_results()
    for r in res:
        eq = r
        loss = r.get_error()
    return eq, loss

if __name__ == '__main__':
    for i in range(1,8): # choose range of index
        case = str(i)
        exec(open("../GVAE/examples/odeformer" + case + ".py").read())
        #exec(open("../GVAE/examples/odeformer_add" + str(i) + ".py").read())
        exec(open("../GVAE/examples/odeformer_basic.py").read())
        exec(open("../GVAE/examples/de_base_data.py").read())
        filename = 'Output_explicit_ProGED_ID'+case+'_best.csv'
        with open(filename, 'w', newline='') as csvfile:
            csvwriter = csv.writer(csvfile)
            csvwriter.writerow(['Predicted solution', 'Loss'])
            for _ in range(5):
                eq, loss = main(t_val.squeeze(), u_val.squeeze(), du_dt_NN.detach().squeeze())
                csvwriter.writerow([eq, loss])