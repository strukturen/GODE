# Example from ODEBench from ODEFormer
# ======================================================================================================================
# Karin Yu, 2025
# Discover equation

true_eq = '-diff(u,t)+0.4*u*(1-u/100.0)-0.24*u/(50.0+u)'
t_low = 1.8
t_high = 17.0
ORDER = 1
y0 = [21.1]
FOURIER = False
fs = 6