# Example NLODE1 of Tsoulos and Lagaris (2006)
# ======================================================================================================================
# Karin Yu, 2025
# Discover equation, homogeneous case
# Case: u = t**0.5, u' - 1/(2*u) = 0

true_solution = ['t**0.5']
true_eq = 'diff(u,t)+-1/(2*u)-0'
t_low = 1.
t_high = 4.
ORDER = 1
y0 = [1]
fs = 30
FOURIER = False



