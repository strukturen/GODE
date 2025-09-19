import signal
import symengine as spe
import sympy

# Timeout exception function
class TimeoutException(Exception):
    pass

def timeout_handler(signum, frame):
    raise TimeoutException("Function timed out")

# Function to replace 'C' with values from the tensor
def replace_C_with_values(expression, values):
    if type(expression) != str:
        expression = str(expression)
    # Store modified expressions for each value
    max_num = values.shape[1]
    expressions = []
    j = 0
    for i in range(len(list(expression))):
        if expression[i] == 'C':
            expressions.append(str(values[0,j].item()))
            j += 1
            if j == max_num:
                j = 0
        else:
            expressions.append(expression[i])

    return ''.join(expressions)

def with_timeout(timeout):
    def decorator(func):
        def wrapper(*args, **kwargs):
            # Set the signal handler and an alarm
            signal.signal(signal.SIGALRM, timeout_handler)
            signal.setitimer(signal.ITIMER_REAL, timeout)
            try:
                result = func(*args, **kwargs)
            finally:
                # Disable the timer
                signal.setitimer(signal.ITIMER_REAL, 0)
            return result
        return wrapper
    return decorator


# Function to normalize expressions
# defined for normalizing the expression
u_t = spe.symbols('u_t')
du_dt = spe.symbols('du_dt')
d2u_dt2 = spe.symbols('d2u_dt2')

def normalize_expr(expr):
    derivative_in_denom = False
    normalizer = []
    if '+' or '-' in str(expr):
        if 'd2u_dt2' in str(expr):
            for term in expr.args:
                num, denom = term.as_numer_denom()
                if 'd2u_dt2' in str(term) and not 'd2u_dt2' in str(denom):
                    if 'd2u_dt2' == str(term):
                        normalizer.append(1)
                    else:
                        normalizer.append(term.coeff(d2u_dt2))
                elif 'd2u_dt2' in str(denom):
                    derivative_in_denom = True
                    normalizer.append(normalize_expr(term * d2u_dt2))
        elif 'du_dt' in str(expr):
            for term in expr.args:
                num, denom = term.as_numer_denom()
                if 'du_dt' in str(term) and not 'du_dt' in str(denom):
                    if 'du_dt' == str(term):
                        normalizer.append(1)
                    else:
                        normalizer.append(term.coeff(du_dt))
                elif 'du_dt' in str(denom):
                    derivative_in_denom = True
                    normalizer.append(1 / du_dt)
        else:
            return expr
    else:
        num, denom = expr.as_numer_denom()
        if 'd2u_dt2' in str(num):
            normalizer.append(expr.coeff(d2u_dt2))
        elif 'du_dt' in str(num):
            normalizer.append(expr.coeff(du_dt))
        elif 'd2u_dt2' in str(denom):
            derivative_in_denom = True
            normalizer.append(1 / d2u_dt2)
        elif 'du_dt' in str(denom):
            derivative_in_denom = True
            normalizer.append(1 / du_dt)
    if len(normalizer) > 0:
        normalizer = '+'.join([str(term) for term in normalizer])
        #print('Normalizer', normalizer)
        expr = (expr/spe.sympify(normalizer)).expand()
        #print('Normalized expression:', expr)
    if derivative_in_denom:
        return normalize_expr(expr)
    return expr

def replace_eq_with_1(expression):
    if type(expression) != str:
        expression = str(expression)
    if expression[0] == 'C':
        return '1' + expression[1:]
    elif expression[:2] == '-C':
        return '1' + expression[2:]
    for i in range(2,len(list(expression))-1):
        if expression[i-1:i+1] == '+C':
            return expression[:i]+'1'+expression[i+1:]
        if expression[i-2:i+1] == '+-C':
            return expression[:i-1]+'1'+expression[i+1:]
    return expression

# convert expressions to symbolic
def convert_to_symbolic(expression, u_t = False):
    if type(expression) != str:
        expression = str(expression)
    expr = expression.replace('d2u_dt2', 'diff(diff(u,t),t)')
    expr = expr.replace('du_dt', 'diff(u,t)')
    expr = expr.replace('u_t', 'u')
    if u_t:
        expr = expr.replace('u', 'u(t)')
    return expr

# convert expressions to numerical
def convert_to_numerical(expression):
    if type(expression) != str:
        expression = str(expression)
    if not 'u(t)' in expression:
        expression = expression.replace('u', 'u(t)')
    expr = expression.replace('diff(diff(u(t),t),t)', 'd2u_dt2')
    expr = expr.replace('diff(u(t),t)', 'du_dt')
    expr = expr.replace('u(t)', 'u_t')
    return expr

def check_order(eq, order = 1, type = 'symbolic'):
    # type = 'numerical': u_t, du_dt, d2u_dt2
    # type = 'symbolic': u(t), diff(u(t),t), diff(diff(u(t),t),t)
    if type == 'numerical':
        if not 'du_dt' in str(eq) and not 'd2u_dt2' in str(eq):
            return False
        elif (order == 1 and 'd2u_dt2' in str(eq)) or (order == 2 and not 'd2u_dt2' in str(eq)) or (order == 1 and not 'du_dt' in str(eq)):
            return False
        else:
            return True
    if type == 'symbolic':
        if eq.count('diff') == 0 or eq.count('diff(diff(diff') > 0:
            return False
        if order == 2 and eq.count('diff(diff') == 0:
            return False
        if order == 1 and eq.count('diff(diff') > 0:
            return False
        return True