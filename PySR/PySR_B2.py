import sys
import symengine as spe
import torch
from pysr import PySRRegressor
from utils import *
import csv

random_seed = 0
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
    # 20, 20, 30 for B2 and B3
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
    #denoise = True, # denoise can mess up results!
)

def main(t_val, u_val, du_dt_NN, d2u_dt2_NN=None):
    if d2u_dt2_NN is None:
        X = np.concatenate((np.expand_dims(t_val, axis = 1), u_val.detach().numpy()), axis = 1) #np.expand_dims(t_val, axis = 1) for B2; t_val.detach().numpy() for B3
        y = du_dt_NN.detach().numpy()
    else:
        X = np.concatenate((t_val.detach().numpy(), u_val.detach().numpy(), du_dt_NN.detach().numpy()), axis=1) #np.expand_dims(t_val, axis = 1) for B2; t_val.detach().numpy() for B3
        y = d2u_dt2_NN.detach().numpy()
    model.fit(X, y)
    loss = model.get_best().loss
    eq = model.get_best().equation.replace('x0', 't')
    eq = eq.replace('x1', 'u')
    if d2u_dt2_NN is not None:
        eq = eq.replace('x2', 'diff(u,t)')
    return eq, loss

if __name__ == '__main__':
    for i in range(1,2): # choose index
        case = str(i)
        # benchmark 2
        #exec(open("../GVAE/examples/ode"+case+".py").read())
        #t_val = np.linspace(t_low, t_high + 1 / fs, int((t_high - t_low) * fs + 1))
        # benchmark 3
        exec(open("../GVAE/examples/pendulum.py").read())
        exec(open("../GVAE/examples/de_base_data.py").read())
        #print('True equation loss:', apply_DE(true_eq, EVALUATE = False, order = ORDER))
        #filename = 'Output_B2_pySR_ODE'+case+'_score.csv'
        filename = 'Output_pySR_pendulum_best.csv'
        with open(filename, 'w', newline='') as csvfile:
            csvwriter = csv.writer(csvfile)
            csvwriter.writerow(['Predicted solution', 'Loss'])
            for _ in range(5):
                if ORDER == 1:
                    eq, loss = main(t_val, u_val, du_dt_NN)
                else:
                    eq, loss = main(t_val, u_val, du_dt_NN, d2u_dt2_NN)
                csvwriter.writerow([eq, loss])
