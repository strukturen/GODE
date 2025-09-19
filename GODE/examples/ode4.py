# Example ODE4 of Tsoulos and Lagaris (2006)
# ======================================================================================================================
# Karin Yu, 2025
# Discover equation, homogeneous case

true_solution = ['sin(8*t)'] #, 'sin(7.999*t)', 'sin(8.001*t)'
true_eq = 'diff(diff(u,t),t)+64*u'
t_low = 0.
t_high = 1.
ORDER = 2
y0 = [0, 10]
fs = 100
# use Fourier features
FOURIER = True


