# Example NLODE4 of Tsoulos and Lagaris (2006)
# ======================================================================================================================
# Karin Yu, 2025
# Discover equation, homogeneous case
# Case: u = log(t^2), u'' * u' + 4/t^3 = 0

true_solution = ['log(log(t))']
true_eq = 'diff(diff(u,t),t)*t^2+diff(u,t)*diff(u,t)*t^2--1/log(t)'
t_low = np.exp(1.)
t_high = 2*np.exp(1.)
ORDER = 2
y0 = [0, 1/np.exp(1)]
fs = 35
FOURIER = False



