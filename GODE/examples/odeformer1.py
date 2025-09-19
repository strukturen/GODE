# Example from ODEBench from ODEFormer
# ======================================================================================================================
# Karin Yu, 2025
# Discover equation

true_eq = '-diff(u,t)+0.7/2.31-1/1.2/2.31*u'
t_low = 0.1
t_high = 15.
ORDER = 1
y0 = [10.0]
FOURIER = False
fs = 8

LOSS_PRINT_THRESHOLD = 0.08