from utils import *
#from scipy_solver import *
import numpy as np

k = 0

solutions_check_trivial = ['t', '-2.5*t', 'sin(3*t)', '2*cos(t/4)+t/3']
tol_check = 50
tol_trivial_DE = 0.5

@with_timeout(2)
def get_DE_loss(expr, data, t_val, ret_val = False, order = 2, force = None,**kwargs):
    # DE should be normalized ahead of this function
    """if 't_tensor' in kwargs:
            t_tensor = kwargs['t_tensor']
            u_val = kwargs['u_val']
            du_dt_val = kwargs['du_dt_val']
            d2u_dt2_val = kwargs['d2u_dt2_val']
            tol_check = kwargs['tol_check']"""
    if expr.count('+') < 1 and expr.count('0') > 0:
        return 1e10
    if len(expr) == 0:
        return 1e10
    t = spe.symbols('t')
    expr_num = convert_to_numerical(expr)
    eq = spe.sympify(expr_num)
    loss = 0
    if not '+' in str(eq) and not '-' in str(eq):
        # print('Error 2')
        return 1e10
    d = spe.lambdify((t, u_t, du_dt, d2u_dt2), [eq])
    if order == 1:
        d_val = np.array([d(t_val[i].squeeze(), data[i,0].squeeze(), data[i,1].squeeze(), 0) for i in range(k, t_val.shape[0] - k)])  # [0]
    elif order == 2:
        d_val = np.array([d(t_val[i].squeeze(), data[i,0].squeeze(), data[i,1].squeeze(), data[i,2].squeeze()) for i in range(k, t_val.shape[0] - k)])  # [0]
    if force is not None:
        d_val = d_val - force[k: t_val.shape[0] - k].detach().numpy()
    loss += np.sqrt(np.mean((d_val) ** 2))
    if loss > tol_check:
        if ret_val:
            return loss, d_val
        return loss
    elif np.isnan(loss):
        if ret_val:
            return 1e10, d_val
        return 1e10
    else:  # check trivial solution
        expr1 = expr.replace('diff', 'spe.diff')
        expr1 = expr1.replace('u(t)', 'u')
        expr1 = expr1.replace('exp', 'spe.exp')
        expr1 = expr1.replace('cos', 'spe.cos')
        expr1 = expr1.replace('sin', 'spe.sin')
        expr1 = expr1.replace('log', 'spe.log')
        expr1 = expr1.replace('^', '**')
        means = []
        for s in solutions_check_trivial:
            u = spe.sympify(s)
            eq = eval(expr1)
            d_check = spe.lambdify(t, [eq])
            d_val_check = d_check(t_val[k:t_val.shape[0] - k])
            if force is not None:
                d_val_check = d_val_check - force[k:t_val.shape[0] - k].detach().numpy()
            means.append(abs(np.mean(d_val_check)))
        if np.mean(means) < tol_trivial_DE or np.mean(means) > 1e6:
            return 1e10
    if ret_val:
        return loss, d_val
    return loss

@with_timeout(2)
def solve_smse_DE(expr, data, t_val, order, y0, norm_expr = None,force = None,**kwargs):
    # solve ODE and get standardized mean square error
    if type(expr) != str:
        expr = str(expr)
    if len(expr) == 0:
        return 1e10
    num_sample = data.shape[0]
    fs = 1 / ((t_val[-1].item() - t_val[0].item()) / (num_sample - 1))
    # currently solve_ode cannot solve all implicit equations but might still yield seem successful --> treat all as implicit
    sol = solve_implicit_ode(expr, t_val[0].item(), t_val[-1].item(), fs=fs, y0=y0, order=order, CHECK_SUCCESS=False, force = force)
    loss = 0
    #added relative mean squared error
    loss += np.mean((sol[1] - data[:,0]) ** 2)/np.mean((data[:,0])**2) # u
    loss += np.mean((sol[2] - data[:,1]) ** 2)/np.mean((data[:,1])**2)  # du/dt
    trivial_loss = np.mean(sol[1][int(num_sample/5):] ** 2)+np.mean(sol[2][int(num_sample/5):] ** 2)
    if order == 2:
        trivial_loss += np.mean(sol[3][int(num_sample/5):]**2)
        loss += np.mean((sol[3] - data[:,2]) ** 2)/np.mean((data[:,2])**2)  # d2u/dt2
    if np.sqrt(trivial_loss) < 0.1:
        return 1e10
    return loss

# complexity function
weights = {
    "constant": 1,   # Weight for constants
    "variable": 1,   # Weight for variables
    "operation": 1,  # Weight for operations
}
def compute_complexity(expr):
    if expr.is_Atom:  # Leaf nodes
        if expr.is_Number:
            return weights["constant"]
        elif expr.is_Symbol:
            return weights["variable"]
    # Operations or composite expressions
    return weights["operation"] + sum(compute_complexity(arg) for arg in expr.args)