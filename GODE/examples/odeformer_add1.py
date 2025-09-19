# Example from ODEBench from ODEFormer
# ======================================================================================================================
# Karin Yu, 2025
# Discover equation

print('This is an additional example outside of ODEBench')

true_eq = '-diff(u,t)+0.21-sin(0.2*t*u)'
t_low = 2.9
t_high = 13.8
ORDER = 1
y0 = [-1.89]
FOURIER = False
fs = 12