# Example from ODEBench from ODEFormer
# ======================================================================================================================
# Karin Yu, 2025
# Discover equation

print('This is an additional example outside of ODEBench')

true_eq = '-diff(u,t)+0.0837*u+log(7.293*u)'
t_low = 1.8
t_high = 8.27
ORDER = 1
y0 = [0.3]
FOURIER = False
fs = 9