# Example ODE5 of Tsoulos and Lagaris (2006)
# ======================================================================================================================
# Karin Yu, 2025
# Discover equation, homogeneous case

true_solution = ['2*t*exp(3*t)']
true_eq = 'diff(diff(u,t),t)-6*diff(u,t)+9*u'
t_low = 0.
t_high = 1.
ORDER = 2
y0 = [0,2]
fs = 100


