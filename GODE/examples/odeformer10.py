# Example from ODEBench from ODEFormer
# ======================================================================================================================
# Karin Yu, 2025
# Discover equation

print('The symbolic equation is out of the distribution of the GVAE.')

true_eq = '-diff(u,t)+0.2*u^1.2*(1-u)-u*(1-0.2)*(1-u)^1.2'
t_low = 1.1
t_high = 14.3
ORDER = 1
y0 = [0.83]
FOURIER = False
fs = 10

LOSS_PRINT_THRESHOLD = 0.08