# Example from ODEBench from ODEFormer
# ======================================================================================================================
# Karin Yu, 2025
# Discover equation

true_eq = '-diff(u,t)+0.0981*(9.7*cos(u)-1)*sin(u)'
t_low = 0.1
t_high = 18.9
ORDER = 1
y0 = [2.4]
FOURIER = False
fs = 10