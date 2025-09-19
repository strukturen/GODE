# Example ODE2 of Tsoulos and Lagaris (2006)
# ======================================================================================================================
# Karin Yu, 2025
# Discover equation, non-homogeneous case

true_solution = ['(t+2)/sin(t)']
true_eq = 'diff(u,t)+u*cos(t)/sin(t)-1/sin(t)'
t_low = 0.1
t_high = 1.
ORDER = 1
y0 = [2.1/np.sin(0.1)]
fs = 50
# use Fourier features
FOURIER = True

