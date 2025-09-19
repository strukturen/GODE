# Scipy solving 1st and 2nd-order ODEs
# ======================================================================================================================
# Karin Yu, 2024

import numpy as np
from scipy.integrate import solve_ivp, odeint
from scipy.interpolate import interp1d
import symengine as spe
import sympy
import sys
from utils import *
import copy

# Need to adjust time wrapper for silver box example!
@with_timeout(2)
def solve_implicit_ode(expr, t_low, t_high, fs, y0, order = 1, CHECK_SUCCESS = False, force = None):
    if type(expr) != str:
        expr = str(expr)
        expr.replace('Derivative', 'diff')
    t_span = (t_low, t_high)  # Time interval
    t_eval = np.linspace(*t_span, int((t_high - t_low+1/fs) * fs))  # Time points to evaluate

    t = spe.symbols('t')
    u = spe.symbols('u')
    v = spe.symbols('v')

    if order == 1:
        expr = expr.replace('diff(u,t)', 'v')
        eq = sympy.sympify(expr)
        eq_function = sympy.lambdify((t, u, v), [eq], modules='numpy')
    elif order == 2:
        w = spe.symbols('w')
        expr = expr.replace('diff(diff(u,t),t)', 'w')
        expr = expr.replace('diff(u,t)', 'v')
        eq = sympy.sympify(expr)
        eq_function = sympy.lambdify((t, u, v, w), [eq], modules='numpy')

    # scipy interp1d
    if force is not None:
        get_force = interp1d(t_eval, force, kind='cubic', fill_value='extrapolate')

    # Newton is about 30% faster than fsolve
    def newton(f, x0, tol=1e-5, max_iter=5):
        x = x0
        for _ in range(max_iter):
            fx = f(x)
            dfx = (f(x + 1e-8) - fx) / 1e-8  # Numerical derivative
            x = x - fx / dfx
            if abs(fx) < tol:
                break
        return x

    def eq_func(t, y):
        if order == 1:
            def algebraic_eqn(v):
                if force is not None:
                    return eq_function(t, y, v)[0] - get_force(t)
                return eq_function(t, y, v)[0]
            v_guess = np.array([1.0])
            v = newton(algebraic_eqn, v_guess)
            return v
        elif order == 2:
            x, x2 = y
            def algebraic_eqn(w):
                if force is not None:
                    return eq_function(t, x, x2, w)[0] - get_force(t)
                return eq_function(t, x, x2, w)[0]
            dxdt = x2
            w_guess = np.array([1.0])
            dvdt = newton(algebraic_eqn, w_guess)
            return [dxdt, dvdt[0]]

    # Solve the ODE
    sol = solve_ivp(eq_func, t_span, y0, method = 'BDF', max_step=0.02,t_eval=t_eval)
    #print('Successfully solved ODE!', sol.status)
    if CHECK_SUCCESS and sol.status != 0:
        raise ValueError('ODE solver failed to converge!')

    if order == 1:
        du_dt_scipy = []
        for i in range(len(sol.t)):
            du_dt_scipy.append(eq_func(sol.t[i], sol.y[:, i])[0])
        du_dt_scipy = np.array(du_dt_scipy)
        return [sol.t, sol.y, du_dt_scipy]
    elif order == 2:
        d2u_dt2_scipy = []
        for i in range(len(sol.t)):
            d2u_dt2_scipy.append(eq_func(sol.t[i], sol.y[:,i])[1])
        d2u_dt2_scipy = np.array(d2u_dt2_scipy)
        return [sol.t, sol.y[0], sol.y[1], d2u_dt2_scipy]

@with_timeout(2)
def solve_ode(expr, t_low, t_high, fs, y0, order = 1, CHECK_SUCCESS = False, method = 'RK45', normalized = False):
    # only works if it can be put into either u' = F(u) or u'' = F(u, u')
    t_span = (t_low, t_high)  # Time interval
    t_eval = np.linspace(*t_span, int((t_high - t_low+1/fs) * fs))  # Time points to evaluate
    t = spe.symbols('t')
    u = spe.symbols('u')
    # normalize expression
    if not normalized:
        expr = expr.replace('u', 'u(t)')
        expr3 = convert_to_numerical(expr)
        norm_expr = str(normalize_expr(spe.sympify(expr3).expand()))
    else:
        norm_expr = convert_to_numerical(expr)
    if order == 1:
        expr = str(spe.sympify('-1*('+norm_expr+'-du_dt)').expand()) # get du_dt = ...
        expr = expr.replace('du_dt', 'diff(u,t)')
    elif order == 2:
        v = spe.symbols('v')
        expr = str(spe.sympify('-1*('+norm_expr + '-d2u_dt2)').expand()) # get d2u_dt2 = ...
        expr = expr.replace('d2u_dt2', 'diff(diff(u,t),t)')
        expr = expr.replace('du_dt', 'v')
    # return to spe expression
    expr = expr.replace('u_t', 'u')
    expr1 = expr.replace('diff', 'Derivative')
    eq = sympy.simplify(sympy.sympify(expr1))
    #print(eq)
    if order == 1:
        eq = sympy.lambdify((t, u), [eq], modules='numpy')
    elif order == 2:
        eq = sympy.lambdify((t, u, v), [eq], modules='numpy')
    def eq_func(t, y):
        if order == 1:
            dxdt = eq(t, y)
            return dxdt
        elif order == 2:
            x, x2 = y
            dxdt = x2
            dvdt = eq(t, x, x2)[0]
            return np.array([dxdt, dvdt])
    # Solve the ODE
    sol = solve_ivp(eq_func, t_span, y0, t_eval=t_eval, method = method, dense_output=True)  #method="LSODA"
    print('Successfully solved ODE!', sol.status)
    if CHECK_SUCCESS and sol.status != 0:
        raise ValueError('ODE solver failed to converge!')

    if order == 1:
        du_dt_scipy = np.array(eq_func(t_eval, sol.y)).flatten()
        if du_dt_scipy.shape[0] == 1:
            du_dt_scipy = np.ones_like(sol.y[0])*du_dt_scipy
        return [sol.t, sol.y.squeeze(), du_dt_scipy]
    elif order == 2:
        d2u_dt2_scipy = np.array(eq_func(t_eval, sol.y))[1,:].flatten()
        if d2u_dt2_scipy.shape[0] == 1:
            d2u_dt2_scipy = np.ones_like(sol.y[0])*d2u_dt2_scipy
        return [sol.t, sol.y[0], sol.y[1], d2u_dt2_scipy]

@with_timeout(2)
def solve_implicit_odeint(expr, t_low, t_high, fs, y0, order = 1, CHECK_SUCCESS = False):
    t_span = (t_low, t_high)  # Time interval
    t_eval = np.linspace(*t_span, int((t_high - t_low+1/dt) * fs))  # Time points to evaluate

    t = spe.symbols('t')
    u = spe.symbols('u')
    v = spe.symbols('v')

    if order == 1:
        expr = expr.replace('diff(u,t)', 'v')
        eq = sympy.sympify(expr)
        eq_function = sympy.lambdify((t, u, v), [eq], modules='numpy')
    elif order == 2:
        w = spe.symbols('w')
        expr = expr.replace('diff(diff(u,t),t)', 'w')
        expr = expr.replace('diff(u,t)', 'v')
        eq = sympy.sympify(expr)
        eq_function = sympy.lambdify((t, u, v, w), [eq], modules='numpy')

    #print(expr)

    def eq_func(y, t):
        if order == 1:
            def algebraic_eqn(v):
                return eq_function(t, y, v)[0]
            v_guess = 1.
            #v = fsolve(algebraic_eqn, v_guess)[0]
            v, infodict, ier, msg = fsolve(algebraic_eqn, v_guess, full_output=True)
            if ier != 1:  # If fsolve didn't converge, raise an error
                raise ValueError(f"fsolve failed to converge: {msg}")
            return v
        elif order == 2:
            x, x2 = y
            def algebraic_eqn(w):
                return eq_function(t, x, x2, w)[0]
            dxdt = x2
            w_guess = 1.
            #dvdt = fsolve(algebraic_eqn, w_guess)[0]
            dvdt, infodict, ier, msg = fsolve(algebraic_eqn, w_guess, full_output=True)
            if ier != 1:  # If fsolve didn't converge, raise an error
                raise ValueError(f"fsolve failed to converge: {msg}")
            #print(t, dvdt[0])
            return [dxdt, dvdt[0]]

    # Solve the ODE
    sol = odeint(eq_func, y0, t_eval)  #, dense_output=True ,  method='BDF',
    #print('Successfully solved ODE!', sol.status)
    if CHECK_SUCCESS and sol.status != 0:
        raise ValueError('ODE solver failed to converge!')

    if order == 1:
        #du_dt_scipy = np.array(eq_func(t_eval, sol.y)).flatten()
        du_dt_scipy = []
        for i in range(t_eval.shape[0]):
            du_dt_scipy.append(eq_func(t_eval[i], sol[i, :])[1])
        du_dt_scipy = np.array(du_dt_scipy)
        return [t_eval, sol.squeeze(), du_dt_scipy]
    elif order == 2:
        d2u_dt2_scipy = []
        for i in range(t_eval.shape[0]):
            d2u_dt2_scipy.append(eq_func(t_eval[i], sol[i,:])[1])
        d2u_dt2_scipy = np.array(d2u_dt2_scipy)
        return [t_eval, sol[:,0], sol[:,1], d2u_dt2_scipy]

@with_timeout(2)
def solve_odeint(expr, t_low, t_high, fs, y0, order = 1, CHECK_SUCCESS = False):
    # only works if it can be put into either u' = F(u) or u'' = F(u, u')
    t_span = (t_low, t_high)  # Time interval
    t_eval = np.linspace(*t_span, int((t_high - t_low+1/fs) * fs))  # Time points to evaluate
    t = spe.symbols('t')
    u = spe.symbols('u')
    # normalize expression
    expr = expr.replace('u', 'u(t)')
    expr3 = expr.replace('diff(diff(u(t),t),t)', 'd2u_dt2')
    expr3 = expr3.replace('diff(u(t),t)', 'du_dt')
    expr3 = expr3.replace('u(t)', 'u_t')
    if order == 1:
        expr = str(spe.sympify(
            '-1*(' + str(normalize_expr(spe.sympify(expr3).expand())) + '-du_dt)').expand())  # get du_dt = ...
        expr = expr.replace('du_dt', 'diff(u,t)')
    elif order == 2:
        v = spe.symbols('v')
        expr = str(normalize_expr(spe.sympify(expr3).expand()))
        # print(expr)
        expr = str(spe.sympify('-1*(' + expr + '-d2u_dt2)').expand())  # get d2u_dt2 = ...
        expr = expr.replace('d2u_dt2', 'diff(diff(u,t),t)')
        expr = expr.replace('du_dt', 'v')
    # return to spe expression
    expr = expr.replace('u_t', 'u')
    expr1 = expr.replace('diff', 'Derivative')
    eq = sympy.simplify(sympy.sympify(expr1))
    if order == 1:
        eq = sympy.lambdify((t, u), [eq], modules='numpy')
    elif order == 2:
        eq = sympy.lambdify((t, u, v), [eq], modules='numpy')
    def eq_func(y, t):
        if order == 1:
            dxdt = eq(t, y)[0]
            return dxdt
        elif order == 2:
            x, x2 = y
            #dxdt = x2
            #dvdt = eq(t, x, x2)[0]
            dvdt = [x2, eq(t, x, x2)[0]]
            return dvdt
    # Solve the ODE
    sol = odeint(eq_func, y0, t_eval)
    #print('Successfully solved ODE!', sol.status)
    if CHECK_SUCCESS and sol.status != 0:
        raise ValueError('ODE solver failed to converge!')

    if order == 1:
        du_dt_scipy = np.array(eq_func(sol.squeeze(),t_eval)).flatten()
        if du_dt_scipy.shape[0] == 1:
            du_dt_scipy = np.ones_like(sol.squeeze())*du_dt_scipy
        return [t_eval, sol.squeeze(), du_dt_scipy]
    elif order == 2:
        d2u_dt2_scipy = np.array(eq_func((sol[:,0],sol[:,1]),t_eval))[1,:].flatten()
        if d2u_dt2_scipy.shape[0] == 1:
            d2u_dt2_scipy = np.ones_like(sol[:,0])*d2u_dt2_scipy
        return [t_eval, sol[:,0], sol[:,1], d2u_dt2_scipy]