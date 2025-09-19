# Example ODE1 of Tsoulos and Lagaris (2006)
# ======================================================================================================================
# Karin Yu, 2025
# Discover equation, non-homogeneous case

true_solution = ['t+2/t']
true_eq = 'diff(u,t)+u/t-2'
t_low = 0.1
t_high = 1.
ORDER = 1
FOURIER = False
fs = 50
y0 = [20.1]


