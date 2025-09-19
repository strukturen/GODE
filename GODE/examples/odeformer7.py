# Example from ODEBench from ODEFormer
# ======================================================================================================================
# Karin Yu, 2025
# Discover equation

true_eq = '-diff(u,t)+0.032*u*log(2.29*u)'
t_low = 1.5
t_high = 12.9
ORDER = 1
y0 = [1.73]
FOURIER = False
fs = 10