# Example from ODEBench from ODEFormer
# ======================================================================================================================
# Karin Yu, 2025
# Discover equation

true_eq = '-diff(u,t)+1.4+0.4*u^5/(123.0+u^5)-0.89*u'
t_low = 2.8
t_high = 7.9
ORDER = 1
y0 = [3.1]
FOURIER = False
fs = 15