# Example from ODEBench from ODEFormer
# ======================================================================================================================
# Karin Yu, 2025
# Discover equation

true_eq = '-diff(u,t)+1.2-0.2*u-exp(-u)'
t_low = 2.0
t_high = 9.6
ORDER = 1
y0 = [0.0]
FOURIER = False
fs = 18