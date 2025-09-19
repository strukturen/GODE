# Example from ODEBench from ODEFormer
# ======================================================================================================================
# Karin Yu, 2025
# Discover equation

true_eq = '-diff(u,t)+0.78*u*(1-u/81.0)-0.9*u^2/(21.2^2+u^2)'
t_low = 1.4
t_high = 8.2
ORDER = 1
y0 = [2.76]
FOURIER = False
fs = 13