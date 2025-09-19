# Example from ODEBench from ODEFormer
# ======================================================================================================================
# Karin Yu, 2025
# Discover equation

print('This is an additional example outside of ODEBench')

true_eq = '-diff(u,t)-7.928+exp(1.03*u)+cos(0.2*t)'
t_low = 0.8
t_high = 12.2
ORDER = 1
y0 = [0.05]
FOURIER = False
fs = 12