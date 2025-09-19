# Example from ODEBench from ODEFormer
# ======================================================================================================================
# Karin Yu, 2025
# Discover equation

print('This is an additional example outside of ODEBench')

true_eq = '-diff(u,t)+4.21*u*(2.1-0.15*u)+2.1*t*cos(t)'
t_low = 1.23
t_high = 17.02
ORDER = 1
y0 = [0.1]
FOURIER = False
fs = 10