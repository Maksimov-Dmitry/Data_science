import numpy as np

def objective(P, Q, data, rho):
    r = data - Q.dot(P)

    val = np.sum(r ** 2)/2. + rho /2. * (np.sum(Q ** 2) + np.sum(P ** 2))

    grad_P = -np.dot(Q.T,r) + rho * P

    grad_Q = -np.dot(r, P.T) + rho * Q

    return val, grad_P, grad_Q


def objective_Q(P0, Q, data, rho):
    val,_ ,grad_Q = objective(P0, Q, data, rho)

    return (val, grad_Q)


def objective_P(P, Q0, data, rho):
    """
    This function returns two values : 
        -The value of the objective function at Q0 fixed 
        -The value of the gradient of P
    """
    val, grad_P,_ = objective(P, Q0, data, rho)

    return (val, grad_P)