# Experimental example: Silver box
# ======================================================================================================================
# Karin Yu, 2025
# Data from nonlinear dynamics benchmarks
# ODE: m*u'' + c*u' + k*u + a*u^3 = f

import numpy as np
import csv
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import nonlinear_benchmarks
from scipy.signal import savgol_filter
import torch

print('Silver box dataset')

# Load data:
# Keep this part fixed, though you can split the train set further in a train and validation set.
# Do not use the test set to make any decision about the model (parameters, hyperparameters, structure, ...)
train, _ = nonlinear_benchmarks.Silverbox()
#test_multisine, test_arrow_full, test_arrow_no_extrapolation = test
#n = test_multisine.state_initialization_window_length

fs = 1/train.sampling_time # sampling frequency

# only use first 10000 points
force_term = '0'
n_data = 1500
t_low = 0.
t_high = n_data*train.sampling_time
print('t_low, t_high:', t_low, t_high)
case = 'specific'
ORDER = 2
NN_LOSS_DEFINED = True # based on u(t)
FOURIER = True
u_val = torch.from_numpy(train.y)[1:n_data+1].float()
f_val = torch.from_numpy(train.u)[1:n_data+1].float()
window_size = 4
poly_order = 3
# Data preprocessing for velocity and acceleration
du_val = savgol_filter(train.y[:n_data+1], window_size, poly_order, deriv=1, delta=train.sampling_time)
d2u_dt2_val = savgol_filter(train.y[1:n_data+1], window_size, poly_order, deriv=2, delta=train.sampling_time)
y0 = [train.y[0], du_val[0]]  # Initial conditions [x(0), v(0)]
du_dt_val = du_val[1:]

t_span = (t_low, t_high)  # Time interval
t_val = np.linspace(*t_span, int((t_high-t_low)*fs))  # Time points to evaluate
print('Number of data points:', len(t_val))

# we don't use de_base_anything
u_t_NN = u_val.unsqueeze(1)
du_dt_NN = torch.from_numpy(du_dt_val).float().unsqueeze(1)
d2u_dt2_NN = torch.from_numpy(d2u_dt2_val).float().unsqueeze(1)






