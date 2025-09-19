# Example ODE3 of Tsoulos and Lagaris (2006)
# ======================================================================================================================
# Karin Yu, 2025
# Discover equation, non-homogeneous case

true_solution = ['exp(t/-5)*sin(t)']
true_eq = 'diff(u,t)+u/5-exp(t/-5)*cos(t)'
t_low = 0.
t_high = 1.
ORDER = 1
y0 = [0]
fs = 50
# use Fourier features
FOURIER = True