# Applied example: Duffing oscillator
# ======================================================================================================================
# Karin Yu, 2025
# ODE: m*u'' + c*u' + k*u + a*u^3 = f

import numpy as np
import csv

print('Duffing Oscillator')

t_low = 0.
t_high = 30.#30.
# Initial conditions
y0 = [0.0, 1.5]  # [x(0), v(0)] 0, 1.5
sampling_freq = 10 #Hz=1/s
fs = sampling_freq
case = 'specific'
# use Fourier features
FOURIER = True
ORDER = 2

# Duffing oscillator parameters
m = 5 #kg, mass
c = 1 #N/(m/s), damping coefficient
k = 7 #N/m, spring constant
a = 25 #N/m^3, cubic stiffness
w = 2 #rad/s, frequency of external force

true_eq = str(m)+'*diff(diff(u,t),t)+'+str(c)+'*diff(u,t)+'+str(k)+'*u+'+str(a)+'*u**3-cos('+str(w)+'*t)'
force_term = 'cos('+str(w)+'*t)'


# 1. Generate measurement data
# Duffing oscillator equation
def duffing(t, y):
    x, v = y
    dxdt = v
    dvdt = (-c * v - k * x - a * x**3 + np.cos(w * t))/m
    return [dxdt, dvdt]



t_span = (t_low, t_high)  # Time interval
t_val = np.linspace(*t_span, int((t_high-t_low)*sampling_freq))  # Time points to evaluate
# Sampling frequence = 10 Hz, 60s * 10/s = 600

# 1. Obtain data from MATLAB
# Load data from MATLAB
data = np.genfromtxt(example_directory+'DuffingO_MATLAB.csv', delimiter=',')
all_data = np.array(data, dtype = float)
t_csv = all_data[1:,0]
u_csv = all_data[1:,1]
du_dt_csv = all_data[1:,2]
d2u_dt2_csv = all_data[1:,3]
t_val = torch.tensor(t_csv).float().view(-1, 1)
u_val = torch.tensor(u_csv).float().view(-1, 1)
du_dt_val = torch.tensor(du_dt_csv).float().view(-1, 1)
d2u_dt2_val = torch.tensor(d2u_dt2_csv).float().view(-1, 1)

# 2. define loss for NN
NN_LOSS_DEFINED = True
print("Simulated MSE: ", np.mean((m*d2u_dt2_csv+c * du_dt_csv + k * u_csv + a * u_csv**3 - np.cos(w * t_csv))**2))

T = w*2*np.pi
T_ind = int(T/(t_csv[1]-t_csv[0]))

fft_d2u_dt2_val = torch.abs(torch.fft.fft(d2u_dt2_val)[:len(d2u_dt2_val)//2])
if DATA_TYPE == 'data_noisy':
    d2u_dt2_mean = np.mean(d2u_dt2_csv, axis=0)
    d2u_dt2_std = np.sqrt(np.sum((d2u_dt2_csv - d2u_dt2_mean) ** 2 / len(t_val), axis=0))
    print('Noise level:', noise_sigma)
    # measured data points
    d2u_dt2_val_noisy = torch.tensor(d2u_dt2_csv + noise_sigma * np.random.randn(*d2u_dt2_csv.shape) * d2u_dt2_std).float().view(
        -1, 1)





