import sys
import symengine as spe
import torch
from pysr import PySRRegressor
from utils import *
import csv

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

model = PySRRegressor(
    niterations=20,  # < Increase me for better results
    populations = 20,
    population_size=30,
    binary_operators=["+", "*", "/", "-", "^"],
    unary_operators=[
        "cos",
        "exp",
        "sin",
        "log",
        # ^ Custom operator (julia syntax)
    ],
    model_selection='best',
    extra_sympy_mappings={"inv": lambda x: 1 / x},
    # ^ Define operator for SymPy as well
    elementwise_loss="loss(prediction, target) = (target-prediction)^2",
    # ^ Custom loss function (julia syntax)
    denoise = True,
)

def main(u_val, du_dt_NN):
    model.fit(u_val.detach().numpy(), du_dt_NN.detach().numpy())
    loss = model.get_best().loss
    eq = model.get_best().equation.replace('x0', 'u')
    return eq, loss

if __name__ == '__main__':
    for i in range(1,8): # choose range of index
        case = str(i)
        #exec(open("../GVAE/examples/odeformer" + case + ".py").read())
        exec(open("../GVAE/examples/odeformer_add" + str(i) + ".py").read())
        exec(open("../GVAE/examples/odeformer_basic.py").read())
        exec(open("../GVAE/examples/de_base_data.py").read())
        filename = 'Output_explicit_pySR_add_ID'+case+'_best.csv'
        with open(filename, 'w', newline='') as csvfile:
            csvwriter = csv.writer(csvfile)
            csvwriter.writerow(['Predicted solution', 'Loss'])
            for _ in range(5):
                eq, loss = main(u_val, du_dt_NN)
                csvwriter.writerow([eq, loss])
