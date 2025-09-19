# Equation Discovery
# ================================
# Karin Yu, 2024

__author__ = 'Karin Yu'

import torch

from parser.cfg_parser import * # parser.
from model.vae import * # model.
import sys
import os
import numpy as np
import symengine as spe
import sympy
# CMA-ES Original
import cma
from scipy.optimize import minimize
from scipy.stats import norm
import csv

import warnings
import copy
from utils import *
from loss_function import *
from scipy_solver import *


# Suppress all warnings
warnings.filterwarnings("ignore")

# checks encoder and decoder
random_seed = 0 # old 1230
np.random.seed(random_seed)
torch.manual_seed(random_seed)

# choose model
grammar_weights = 'savedmodels/nlode_b3_bigru_d100_3h80_c789_L21_b4_ELU_GC0_bs500_ES1000_41k.model'

LATENT = 21
BETA_KL = 4
if BETA_KL == -1: BETAKL = 1
elif BETA_KL == 0: BETAKL = 0.1
elif BETA_KL == 1: BETAKL = 0.01
elif BETA_KL == 2: BETAKL = 0.001
elif BETA_KL == 3: BETAKL = 0.0001
elif BETA_KL == 4: BETAKL = 0.00001
num_hidden = 3
rnn = 'bigru'
HIDDEN = 80
DENSE = 100
CONV1 = 7
decoder_activation = 'ELU'
Encoder_type = 'conv'
num_data = '_41k'
grammar_file = 'grammar/nlode_b3_gvae.grammar'
MODE = 'cpu'
params ={'num_hidden': num_hidden, 'hidden': HIDDEN, 'dense': DENSE, 'conv1': CONV1, 'conv2': CONV1+1, 'conv3': CONV1+2, 'rnn': rnn, 'encoder_type': Encoder_type, 'decoder_act': decoder_activation, 'mode': MODE}
train_data_file = 'data/data_nlode_b3_datagen'+num_data+'_train.txt'
OPTIMIZER = 'CMAES'
DATA_TYPE = 'data_noisy'

# Depending on which Benchmark is chosen some small modifications need to be made
# TODO: TURN "LOSS"-Function in examples/de_base_data.py on
# TODO: Adjust MAX_LEN in grammar/common_args.py if necessary
# Increase loss time of solve_smse_DE to 5s

# number of runs
if len(sys.argv) > 1:
    num_run = int(sys.argv[1]) # number of runs
else:
    num_run = 1

# Real Life Cases
if DATA_TYPE == 'data':
    noise_sigma = 0
elif DATA_TYPE == 'data_noisy':
    print('Noisy case')
    noise_sigma = 0.05
NN_LOSS_DEFINED = False
FOURIER = False
LOSS_PRINT_THRESHOLD = 0.1
sigma_mse = 1.0
ALPHAIC = 0.1 # usually 0.1
alpha_ic = ALPHAIC
min_alpha, max_alpha = 0.1, 0.9
ode_solve_threshold = 200 # initial threshold for ODE solving
MAXITER = 5 # for CMA-ES
POPSIZE = 500 # for CMA-ES
print('Alpha IC:', alpha_ic)

# load common arguments
exec(open('grammar/common_args.py').read())
example_directory = 'examples/'
# choose problem
# Duffing Oscillator
exec(open("examples/duffing_oscillator.py").read())
# Van der Pol Oscillator
#exec(open("examples/vanderpol_oscillator.py").read())
# Pendulum
#exec(open("examples/pendulum.py").read())

# 1. load grammar and VAE
grammar = Grammar(grammar_file)
grammar_model = GrammarModel(grammar, grammar_weights, latent_dim=LATENT, hypers = params)
grammar_model.vae.encoder.eval()
grammar_model.vae.decoder.eval()

# 2. add attributes
disallowed_dict = {}
grammar.add_disallowed(disallowed_dict)

# 3. define differential equation and boundary conditions
t = spe.symbols('t')

if not 't_val' in globals():
    t_val = np.linspace(t_low, t_high + 1 / fs, int((t_high - t_low) * fs + 1))

# choose type of data
if DATA_TYPE == 'analytical solution':
    exec(open("examples/de_base_analytical.py").read())
elif DATA_TYPE == 'data':
    exec(open("examples/de_base_data.py").read())
elif DATA_TYPE == 'data_noisy':
    exec(open("examples/de_base_data.py").read())

exec(open("loss_function.py").read())

# bias equation
if ORDER == 1:
    biased_eq = 'C*diff(u,t)+C*u'
    data = torch.concat((u_t_NN, du_dt_NN), dim=1).detach().numpy()
if ORDER == 2:
    biased_eq = 'C*diff(diff(u,t),t)+C*u'
    data = torch.concat((u_t_NN, du_dt_NN, d2u_dt2_NN), dim=1).detach().numpy()
k = 0

# Define dictionary so that the same skeleton is only explored ones
skeleton_dict = {}
sorted_loss = list()
complexity_list = list()
mse_list = list()


def find_C_values(C, skeleton_expr_eq, ORDER, ode_solve_threshold, DE_loss=False):
    C_torch = torch.from_numpy(C).unsqueeze(0)
    eq = replace_C_with_values(skeleton_expr_eq, C_torch)
    num_eq = convert_to_numerical(eq)
    norm_eq = spe.sympify(num_eq).expand()
    norm_eq = convert_to_symbolic(str(norm_eq))
    if len(eq) == 0:
        return 1e10
    try:
        # get residual loss of differential equation
        loss = get_DE_loss(norm_eq, data, t_val, order=ORDER)
        if DE_loss:
            return loss
        # print(eq, norm_eq,loss)
        if loss == np.nan or loss == np.inf or loss == -np.inf:
            return 1e10
        # if below a certain threshold, solve ODE
        if loss < ode_solve_threshold:
            more_accurate_loss = solve_smse_DE(eq, data, t_val, order=ORDER, y0=y0, norm_expr=norm_eq)
            if more_accurate_loss == np.nan or more_accurate_loss == np.inf or more_accurate_loss == -np.inf:
                return 1e10
            return more_accurate_loss
    except:
        return 1e10
    return loss

def optimize_constants(skeleton_expr_eq, ORDER, ODE_threshold, C_INPUT = 10):
    find_C = lambda C: find_C_values(C, skeleton_expr_eq, ORDER, ODE_threshold)
    C_initial = np.ones(C_INPUT)
    C_optim = minimize(find_C, C_initial, method = 'Nelder-Mead',options={'fatol': 1e-3, 'maxiter': 500})
    return C_optim.x


# 3. objective function
def obj_func(z, grammar_model, eval = False):
    global sorted_loss
    if len(z.shape) == 1:
        z = torch.from_numpy(z).float().unsqueeze(0)
    else:
        z = torch.from_numpy(z).float()
    # 1. Skeleton expression
    skeleton_decoded = grammar_model.decode(z, with_attr = True, loss = False)
    if len(skeleton_decoded[0]) == 0:
        if eval:
            return 1e10, ''
        return 1e10
    # adjusted due to force_term
    if force_term == '0':
        skeleton_expr_eq = replace_eq_with_1(skeleton_decoded[0])
        if skeleton_expr_eq.count('+') < 1:
            skeleton_dict[skeleton_expr_eq] = (1e10, '')
            if eval:
                return 1e10, ''
            return 1e10
    else:
        skeleton_expr_eq = skeleton_decoded[0]+'-'+force_term
    if skeleton_expr_eq in skeleton_dict:
        mse_loss, eq = skeleton_dict[skeleton_expr_eq]
        if mse_loss < 1e10:
            num_eq = convert_to_numerical(eq)
            norm_eq = spe.sympify(num_eq).expand()
            sym_norm_eq = convert_to_symbolic(norm_eq, u_t=True)
            norm_eq = sympy.sympify(sym_norm_eq)
            complexity = compute_complexity(norm_eq)
            loss = alpha_ic * complexity + (1 - alpha_ic) * data.shape[0] / sigma_mse ** 2 * mse_loss
            if eval:
                return loss, eq
            return loss
        else:
            if eval:
                return 1e10, ''
            return 1e10
    else:
        # check skeleton expression and order
        if not check_order(str(skeleton_expr_eq), order = ORDER, type = 'symbolic'):
            skeleton_dict[skeleton_expr_eq] = (1e10, '')
            if eval:
                return 1e10, ''
            return 1e10
        # 2. find C values
        Num_C = skeleton_expr_eq.count('C')
        C_optim = optimize_constants(skeleton_expr_eq, ORDER, ode_solve_threshold, Num_C)
        mse_loss = find_C_values(C_optim, skeleton_expr_eq, ORDER, ode_solve_threshold)
        if mse_loss < 1e10:
            sorted_loss.append(find_C_values(C_optim, skeleton_expr_eq, ORDER, ode_solve_threshold, DE_loss = True))
        eq = replace_C_with_values(skeleton_expr_eq, torch.from_numpy(C_optim).unsqueeze(0))
    if mse_loss < 1e10:
        num_eq = convert_to_numerical(eq)
        norm_eq = spe.sympify(num_eq).expand()
        sym_norm_eq = convert_to_symbolic(norm_eq, u_t=True)
        norm_eq = sympy.sympify(sym_norm_eq)
        complexity = compute_complexity(norm_eq)
        # adapted IC criterion = alpha*complexity + (1-alpha)*n/sigma^2 * MSE
        loss = alpha_ic*complexity + (1-alpha_ic)*data.shape[0]/sigma_mse**2*mse_loss
        skeleton_dict[skeleton_expr_eq] = (mse_loss,eq)
    else:
        loss = 1e10
        skeleton_dict[skeleton_expr_eq] = (1e10, '')
    if loss < 25*LOSS_PRINT_THRESHOLD:
        print('Loss:', loss, eq)
    if eval:
        return loss, eq
    return loss

print('Loss of true equation: {:.5f}'.format(get_DE_loss(true_eq, data, t_val, order = ORDER)))
if DATA_TYPE != 'analytical solution':
    LOSS_PRINT_THRESHOLD = max(1.5*get_DE_loss(true_eq, data, t_val, order = ORDER), LOSS_PRINT_THRESHOLD)

def main():
    global sorted_loss, complexity_list, mse_list, alpha_ic, ode_solve_threshold
    # 4. optimization
    if OPTIMIZER == 'CMAES':
        print('Optimizer: CMA-ES')
        skeleton_dict = {}
        alpha_ic = ALPHAIC
        # new initialization point:
        x0 = x0_best
        sigma0 = 0.5
        es = cma.CMAEvolutionStrategy(x0, sigma0, {'popsize': POPSIZE, 'ftarget': 1e-3})
        best_sol = None
        ode_solve_threshold = 200
        for iteration in range(MAXITER):
            solutions = es.ask()
            sorted_loss = list()
            complexity_list = list()
            mse_list = list()
            ic_values = [obj_func(s, grammar_model) for s in solutions]
            sorted_indices = [i[0] for i in sorted(enumerate(sorted_loss), key=lambda x: x[1])]
            ode_solve_threshold = 1.05*np.mean([sorted_loss[sorted_indices[i]] for i in range(min(20,len(sorted_indices))) if not np.isnan(sorted_loss[sorted_indices[i]])])
            if np.isnan(ode_solve_threshold):
                ode_solve_threshold = 200
            print('ODE solve threshold:', ode_solve_threshold)
            if best_sol is not None:
                solutions.append(best_sol)
                best_sol_obj = obj_func(best_sol, grammar_model)
                ic_values.append(best_sol_obj)
                if len(sorted_indices)> 0 and best_sol_obj > ic_values[sorted_indices[0]]:
                    best_sol = solutions[sorted_indices[0]]
            else:
                if len(sorted_indices) > 0:
                    best_sol = solutions[sorted_indices[0]]
            sorted_sol_indices = sorted(range(len(ic_values)), key=lambda k: ic_values[k])
            sorted_sol = [solutions[i] for i in sorted_sol_indices[:POPSIZE]]
            sorted_ic_values = [ic_values[i] for i in sorted_sol_indices[:POPSIZE]]
            es.tell(sorted_sol, sorted_ic_values)
            es.disp()
            es.logger.add()
            if es.stop('tolfun'):
                break
        zopt = es.result.xbest
        loss, eq = obj_func(zopt, grammar_model, eval = True)
        print('Predicted loss: ', loss, ' ', eq)
        decoded_sol = eq

    print('Solution:', decoded_sol)
    decoded_changed = decoded_sol.replace('Derivative', 'diff')
    decoded_eq = 1e10
    try:
        mse_loss = solve_smse_DE(decoded_changed, data, t_val, order=ORDER, y0=y0)
        print('Equations - Prediction loss: {:.3f}'.format(mse_loss))
        sym_norm_eq = convert_to_symbolic(decoded_changed, u_t=True)
        norm_eq = sympy.sympify(sym_norm_eq)
        complexity = compute_complexity(norm_eq)
        decoded_eq = alpha_ic*complexity + (1-alpha_ic)*data.shape[0]/sigma_mse**2*mse_loss
        print('Criterion: {:.3f} with complexity {:.1f}'.format(decoded_eq, complexity))
    except:
        pass

    print('True equation:', true_eq)
    true_sol = solve_smse_DE(true_eq, data, t_val, order=ORDER, y0=y0, norm_expr = true_eq)
    print(
        'Equation - True loss: {:.3f}'.format(true_sol))
    try:
        sym_norm_eq = convert_to_symbolic(true_eq, u_t=True)
        norm_eq = sympy.sympify(sym_norm_eq)
        complexity = compute_complexity(norm_eq)
        print('Criterion: {:.3f} with complexity {:.1f}'.format(alpha_ic*complexity + (1-alpha_ic)*data.shape[0]/sigma_mse**2*true_sol, complexity))
    except:
        pass
    return decoded_sol, decoded_eq

if __name__ == '__main__':

    if 'biased_eq' in globals():
        print('Biased equation', biased_eq)
        x0_best = grammar_model.encode([biased_eq]).detach().numpy()
    else:
        x0_best = np.zeros(LATENT)

    print('Check x0_best:', obj_func(x0_best,grammar_model))

    save_name = ('Output_B3_duffing_oscillator.csv')
    print(save_name)
    print(true_eq)

    if save_name in os.listdir():
        with open(save_name, mode='r') as csvfile:
            csv_read = csv.reader(csvfile)
            i = 0
            for _ in csv_read:
                i += 1
        with open(save_name, mode='a', newline='') as csvfile:
            csvwriter = csv.writer(csvfile)
            while i <= num_run:
                print('### Run: ', i)
                pred_sol, pred_eq = main()
                csvwriter.writerow([i, pred_sol, pred_eq])
                i += 1
    else:
        # Save results in a CSV file
        with open(save_name, 'w', newline='') as csvfile:
            csvwriter = csv.writer(csvfile)
            csvwriter.writerow(['Run', 'Predicted solution','Loss equation'])
            for i in range(1, num_run + 1):
                print('### Run: ', i)
                pred_sol, pred_eq = main()
                csvwriter.writerow([i, pred_sol, pred_eq])