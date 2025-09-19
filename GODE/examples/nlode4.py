# Example NLODE --> NLODE4
# ======================================================================================================================
# Karin Yu, 2025
# Discover equation, homogeneous case
# Case: u''(t)^2-9u(t)^2=0

true_solution = ['sin(3*t)']
true_eq = 'diff(diff(u,t),t)^2+-81*u^2-0'
t_low = 0
t_high = 2*np.pi
ORDER = 2
y0 = [0, 3]
fs = 30
FOURIER = True