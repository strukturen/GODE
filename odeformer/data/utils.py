import signal
import symengine as spe

class TimeoutException(Exception):
    pass

def timeout_handler(signum, frame):
    raise TimeoutException("Function timed out")

# Function to replace 'C' with values from the tensor
def replace_C_with_values(expression, values):
    # Store modified expressions for each value
    expressions = []
    j = 0
    for i in range(len(list(expression))):
        if expression[i] == 'C':
            expressions.append(str(values[0,j].item()))
            j += 1
        else:
            expressions.append(expression[i])

    return ''.join(expressions)

def with_timeout(timeout):
    def decorator(func):
        def wrapper(*args, **kwargs):
            # Set the signal handler and an alarm
            signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(timeout)
            try:
                result = func(*args, **kwargs)
            finally:
                # Cancel the alarm
                signal.alarm(0)
            return result
        return wrapper
    return decorator

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