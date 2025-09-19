# Example from ODEBench from ODEFormer
# ======================================================================================================================
# Karin Yu, 2025
# Discover equation

true_eq = '-diff(u,t)+0.1*u+0.04*u^3-0.001*u^5'
t_low = 0.8
t_high = 9.2
ORDER = 1
y0 = [0.94]
FOURIER = False
fs = 18