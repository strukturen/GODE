# Example from ODEBench from ODEFormer
# ======================================================================================================================
# Karin Yu, 2025
# Discover equation

true_eq = '-diff(u,t)+0.1-0.55*u+u^2/(u^2+1)'
t_low = 1.5
t_high = 11.3
ORDER = 1
y0 = [0.002]
FOURIER = False
fs = 16