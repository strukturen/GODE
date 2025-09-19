# Example from ODEBench from ODEFormer
# ======================================================================================================================
# Karin Yu, 2025
# Discover equation

true_eq = '-diff(u,t)+0.4*u*(1-u/100.0)-0.3'
t_low = 1.4
t_high = 15.2
ORDER = 1
y0 = [14.3]
FOURIER = False
fs = 9