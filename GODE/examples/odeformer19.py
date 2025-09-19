# Example from ODEBench from ODEFormer
# ======================================================================================================================
# Karin Yu, 2025
# Discover equation

true_eq = '-diff(u,t)-0.08*u/(0.8+u)+u*(1-u)'
t_low = 2.5
t_high = 8.0
ORDER = 1
y0 = [0.13]
FOURIER = False
fs = 20