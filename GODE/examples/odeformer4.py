# Example from ODEBench from ODEFormer
# ======================================================================================================================
# Karin Yu, 2025
# Discover equation

true_eq = '-diff(u,t)-0.5+1/(1+exp(0.5-u/0.96))'
t_low = 1.5
t_high = 9.
ORDER = 1
y0 = [0.8]
FOURIER = False
fs = 9