import math
import numpy as np

def rk4_integration(f, r, t, h, **kwargs):
    """
    Numeric ODE integration via the 4th-order Runge-Kutta method.
    * f - function that computes dr/dr
    * r - quantity (or vector of quantities) being integrated
    * t - independent integration variable
    * h - stepsize for integration variable
    * **kwargs - additional keyword arguments for f
    """

    k1 = h*f(r, t, **kwargs)
    k2 = h*f(r+k1/2, t+h/2, **kwargs)
    k3 = h*f(r+k2/2, t+h/2, **kwargs)
    k4 = h*f(r+k3, t+h, **kwargs)

    dr = (k1 + 2*k2 + 2*k3 + k4)/6

    if np.isinf(dr).any() or np.isnan(dr).any():
        raise Exception("Cannot update with NaN or inf values")
    if (r + dr <= 0.0).any():
        raise Exception("Values must be positive")

    return r + dr
