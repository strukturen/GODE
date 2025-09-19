# Example NLODE --> NLODE5
# ======================================================================================================================
# Karin Yu, 2025
# Discover equation, homogeneous case
# Case: u''(t)*u'(t)-2.1*u(t)-9.84*t^3 = 0

true_solution = ['0.8*t^3']
true_eq = 'diff(diff(u,t),t)*diff(u,t)+-2.1*u-9.84*t^3'
t_low = 0
t_high = 2.5
ORDER = 2
y0 = [0, 0]
fs = 50
FOURIER = True