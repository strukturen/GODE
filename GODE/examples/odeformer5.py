# Example from ODEBench from ODEFormer
# ======================================================================================================================
# Karin Yu, 2025
# Discover equation

true_eq = '-diff(u,t)+9.81-0.0021175*u^2'
t_low = 2
t_high = 23.
ORDER = 1
y0 = [0.5]
FOURIER = False
fs = 6