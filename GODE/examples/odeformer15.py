# Example from ODEBench from ODEFormer
# ======================================================================================================================
# Karin Yu, 2025
# Discover equation

true_eq = '-diff(u,t)+0.4*u*(1-u/95.)-u^2/(1+u^2)'
t_low = 3.8
t_high = 13.8
ORDER = 1
y0 = [44.3]
FOURIER = False
fs = 9