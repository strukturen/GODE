# Example from ODEBench from ODEFormer
# ======================================================================================================================
# Karin Yu, 2025
# Discover equation

true_eq = '-diff(u,t)+0.14*u*(u/4.4-1)*(1-u/130.0)'
t_low = 1.8
t_high = 9.1
ORDER = 1
y0 = [6.123]
FOURIER = False
fs = 15