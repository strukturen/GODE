# Example from ODEBench from ODEFormer
# ======================================================================================================================
# Karin Yu, 2025
# Discover equation

print('This is an additional example outside of ODEBench')

true_eq = '-diff(u,t)+0.1137*u-9.7*sin(1.32*t)'
t_low = 3.52
t_high = 21.87
ORDER = 1
y0 = [12.3]
FOURIER = False
fs = 9