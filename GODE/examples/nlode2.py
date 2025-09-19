# Example NLODE3 of Tsoulos and Lagaris (2006)
# ======================================================================================================================
# Karin Yu, 2025
# Discover equation, homogeneous case
# Case: u = log(t^2), u'' * u' + 4/t^3 = 0

true_solution = ['log(t^2)']
true_eq = 'diff(diff(u,t),t)*diff(u,t)*1--4/t^3'
t_low = 1.
t_high = 2.
ORDER = 2
y0 = [0,1]
fs = 50
FOURIER = False



