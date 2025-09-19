# Example from ODEBench from ODEFormer
# ======================================================================================================================
# Karin Yu, 2025
# Discover equation

print('This is an additional example outside of ODEBench')

true_eq = '-diff(u,t)-0.9832*cos(0.132*t)-u'
t_low = 0.9
t_high = 52.8
ORDER = 1
y0 = [1.89]
FOURIER = False
fs = 4