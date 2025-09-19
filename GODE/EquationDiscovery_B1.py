# Equation Discovery
# ================================
# Karin Yu, 2025

__author__ = 'Karin Yu'

from parser.cfg_parser import * # parser.
from model.vae import * # model.
import sys
import numpy as np
import symengine as spe
# CMA-ES Original
import cma
from scipy.optimize import minimize
from scipy.stats import norm
import csv

import warnings
import copy
from utils import *
from loss_function import *


# Suppress all warnings
warnings.filterwarnings("ignore")

# checks encoder and decoder
random_seed = 0 # old 1230
np.random.seed(random_seed)
torch.manual_seed(random_seed)

# choose model
grammar_weights = 'savedmodels/nlode_b1_explicit_bigru_d80_3h80_c789_L24_b3_ELU_GC0_bs500_ES1000_11k.model'

LATENT = 24
BETA_KL = 3
if BETA_KL == -1: BETAKL = 1
elif BETA_KL == 0: BETAKL = 0.1
elif BETA_KL == 1: BETAKL = 0.01
elif BETA_KL == 2: BETAKL = 0.001
elif BETA_KL == 3: BETAKL = 0.0001
elif BETA_KL == 4: BETAKL = 0.00001
num_hidden = 3
rnn = 'bigru'
HIDDEN = 80
DENSE = 80
MODE = 'bigpu'
CONV_HIDDEN = 0 # multiplied by two
CONV1 = 7
C_INPUT = 10
N_Beta = 1
Loss_type = 'recon'
Encoder_type = 'conv'
conv_type = 'kernel'
EARLYSTOPPING = 1000
MODE = 'cpu'
decoder_activation = 'ELU'
params ={'num_hidden': num_hidden, 'hidden': HIDDEN, 'dense': DENSE, 'conv1': CONV1, 'conv2': CONV1+1, 'conv3': CONV1+2, 'rnn': rnn, 'encoder_type': Encoder_type, 'decoder_act': decoder_activation, 'mode': MODE}
grammar_file = 'grammar/nlode_b1.grammar'
train_data_file = 'data/data_nlode_b1_11k_train.txt'
OPTIMIZER = 'CMAES'
DATA_TYPE = 'data_noisy'

# TODO: TURN "LOSS"-Function in examples/de_base_data.py off

# choose problem
if len(sys.argv) > 1:
    case = str(sys.argv[1]) # defined in command line (e.g. python3 SolutionSolver.py 3) for case = '3'
    print('Case ', case)
    num_run = int(sys.argv[2]) # number of runs
else:
    case = '2' # self defined
    num_run = 1
NN_LOSS_DEFINED = True
FOURIER = False
LOSS_PRINT_THRESHOLD = 0.1
MAXITER = 10 # for CMA-ES
POPSIZE = 100 # for CMA-ES
sigma_mse = 1
alpha_ic = 0.1

# load common arguments
exec(open('grammar/common_args.py').read())

t = spe.symbols('t')
exec(open("scipy_solver.py").read())

# Cases
if DATA_TYPE == 'data':
    noise_sigma = 0
elif DATA_TYPE == 'data_noisy':
    print('Noisy case')
    noise_sigma = 0.05
#exec(open("examples/odeformer"+case+".py").read())
exec(open("examples/odeformer_add"+case+".py").read())
exec(open("examples/odeformer_basic.py").read())

print('max_len:', MAX_LEN)

# 1. load grammar and VAE
grammar = Grammar(grammar_file)
grammar_model = GrammarModel(grammar, grammar_weights, latent_dim=LATENT, hypers = params, mode = 'cpu')
num_trainable_params_enc = sum(p.numel() for p in grammar_model.vae.encoder.parameters() if p.requires_grad)
num_trainable_params_dec = sum(p.numel() for p in grammar_model.vae.decoder.parameters() if p.requires_grad)
print(f'Number of trainable parameters: {num_trainable_params_enc}, {num_trainable_params_dec}')

grammar_model.vae.encoder.eval()
grammar_model.vae.decoder.eval()

# 2. add attributes
disallowed_dict = {}
grammar.add_disallowed(disallowed_dict)

# 3. define differential equation and boundary conditions
t = spe.symbols('t')

if DATA_TYPE == 'analytical solution':
    n_sample = 50
elif DATA_TYPE == 'data':
    n_sample = 100
elif DATA_TYPE == 'data_noisy':
    n_sample = 200

if not 't_val' in globals():
    t_val = np.linspace(t_low, t_high, n_sample)

if DATA_TYPE == 'analytical solution':
    exec(open("examples/de_base_analytical.py").read())
elif DATA_TYPE == 'data':
    exec(open("examples/de_base_data.py").read())
elif DATA_TYPE == 'data_noisy':
    exec(open("examples/de_base_data.py").read())

if ORDER == 1:
    data = torch.concat((u_t_NN, du_dt_NN), dim=1).detach().numpy()

# Function to replace 'C' with values from the tensor
def replace_C_with_values(expression, values):
    # Store modified expressions for each value
    expressions = []
    j = 0
    for i in range(len(list(expression))):
        if expression[i] == 'C':
            expressions.append(str(values[0,j].item()))
            j += 1
            if j > 8:
                return '10000000000'
        else:
            expressions.append(expression[i])

    return ''.join(expressions)


# Define dictionary so that the same skeleton is only explored ones
skeleton_dict = {}

def find_C_values(C, skeleton_expr_eq, ORDER, DE_loss=False):
    C_torch = torch.from_numpy(C).unsqueeze(0)
    eq = replace_C_with_values('-diff(u,t)+'+skeleton_expr_eq, C_torch)
    if len(eq) == 0:
        return 1e10
    try:
        # get residual loss of differential equation
        loss = get_DE_loss(eq, data, t_val, order=ORDER)
        if loss == np.nan or loss == np.inf or loss == -np.inf:
            return 1e10
    except:
        return 1e10
    return loss

def optimize_constants(skeleton_expr_eq, ORDER, C_INPUT = 10):
    find_C = lambda C: find_C_values(C, skeleton_expr_eq, ORDER)
    C_initial = np.ones(C_INPUT)
    C_optim = minimize(find_C, C_initial, method = 'Nelder-Mead',options={'fatol': 1e-3, 'xatol': 1e-3, 'maxiter': 300})
    return C_optim.x

# 3. objective function
@with_timeout(2) # throw exception after 2 second2
def obj_func(z, grammar_model, eval = False):
    if len(z.shape) == 1:
        z = torch.from_numpy(z).float().unsqueeze(0)
    else:
        z = torch.from_numpy(z).float()
    decoded = grammar_model.decode(z, with_attr = True, loss = False)
    if decoded[0] in skeleton_dict:
        mse_loss, eq = skeleton_dict[decoded[0]]
    else:
        num_C = decoded[0].count('C')
        C_optim = optimize_constants(decoded[0], ORDER)
        mse_loss = find_C_values(C_optim, decoded[0], ORDER, num_C)
        eq = replace_C_with_values('-diff(u,t)+'+decoded[0], torch.from_numpy(C_optim).unsqueeze(0))
    if mse_loss >= 9e9:
        if eval:
            return mse_loss, eq
        return mse_loss
    norm_eq = sympy.sympify(eq)
    complexity = compute_complexity(norm_eq)
    loss = alpha_ic*complexity + (1-alpha_ic)*data.shape[0]/sigma_mse**2*mse_loss
    skeleton_dict[decoded[0]] = (mse_loss, eq)
    if loss < LOSS_PRINT_THRESHOLD:
        print('Loss:', loss, eq)
    if eval:
        return loss, eq
    return loss

print('True equation:', true_eq)
print('Loss of true equation: {:.5f}'.format(get_DE_loss(true_eq, data, t_val, order=ORDER)))
if DATA_TYPE != 'analytical solution':
    LOSS_PRINT_THRESHOLD = max(1.5*get_DE_loss(true_eq, data, t_val, order=ORDER), LOSS_PRINT_THRESHOLD)
    if LOSS_PRINT_THRESHOLD > 1e3:
        LOSS_PRINT_THRESHOLD = 1

def main():
    # 4. optimization
    if OPTIMIZER == 'CMAES':
        print('Optimizer: CMA-ES')
        # sample a batch
        x0 = x0_best
        sigma0 = 0.5
        es = cma.CMAEvolutionStrategy(x0, sigma0, {'popsize': POPSIZE, 'ftarget': 1e-3})
        for iteration in range(MAXITER):
            # print('### Iteration ', iteration)
            solutions = es.ask()
            sorted_loss = list()
            ic_values = [obj_func(s, grammar_model) for s in solutions]
            sorted_sol_indices = sorted(range(len(ic_values)), key=lambda k: ic_values[k])
            sorted_sol = [solutions[i] for i in sorted_sol_indices[:POPSIZE]]
            sorted_ic_values = [ic_values[i] for i in sorted_sol_indices[:POPSIZE]]
            es.tell(sorted_sol, sorted_ic_values)
            es.disp()
            es.logger.add()
            if es.stop('tolfun'):
                break
        zopt = es.result.xbest
        loss, eq = obj_func(zopt, grammar_model, eval=True)
        print('Predicted loss: ', loss, ' ', eq)
        decoded_sol = eq

    print('Solution:', decoded_sol)
    decoded_changed = decoded_sol.replace('Derivative', 'diff')
    decoded_eq = get_DE_loss(decoded_changed, data, t_val, order=ORDER)
    print('Equations - Prediction loss: {:.3f}'.format(decoded_eq))

    print('True equation:', true_eq)
    print(
        'Equation - True loss: {:.3f}'.format(get_DE_loss(true_eq, data, t_val, order=ORDER)))
    return decoded_sol, decoded_eq

if __name__ == '__main__':

    if 'biased_eq' in globals():
        print('Biased equation', biased_eq)
        x0_best = grammar_model.encode([biased_eq]).detach().numpy()
    else:
        x0_best = np.zeros(LATENT)

    print('Check x0_best:', obj_func(x0_best,grammar_model))

    # Save results in a CSV file
    with open('Output_explicit_addID'+case+'.csv', 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(['Run', 'Predicted solution','Loss equation'])
        for i in range(1, num_run + 1):
            print('### Run: ', i)
            pred_sol, pred_eq = main()
            csvwriter.writerow([i, pred_sol, pred_eq])
