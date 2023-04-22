import math
import numpy as np

class DataContainer:
    pass

consts = DataContainer()

consts.A = {
    "H": 1.008, "He": 4.0026, "Li": 6.94, "Be": 9.0122, "B": 10.81, "C": 12.011, "N": 14.007, "O": 15.999, "F": 18.998, "Ne": 20.180, "Na": 22.990, "Mg": 24.305, "Al": 26.982, "Si": 28.085, "P": 30.974, "S": 32.06, "Cl": 35.45,
    "Ar": 39.948, "K": 39.098, "Ca": 40.078, "Sc": 44.956, "Ti": 47.867, "V": 50.942, "Cr": 51.996, "Mn": 54.938, "Fe": 55.845,
    "Z": 15.5 # Average of solar composition for metals -- catch-all for species with undefined mass fractions
}

consts.Z = {
    "H": 1, "He": 2, "Li": 3, "Be": 4, "B": 5, "C": 6, "N": 7, "O": 8, "F": 9, "Ne": 10, "Na": 11, "Mg": 12, "Al": 13, "Si": 14, "P": 15, "S": 16, "Cl": 17, "Ar": 18, "K": 19, "Ca": 20, "Sc": 21, "Ti": 22, "V": 23, "Cr": 24,
    "Mn": 25, "Fe": 26,
    "Z": 15.5/2 # Average of solar composition for metals -- catch-all for species with undefined mass fractions
}

consts.m_u = 1.660538782e-27
consts.k_B = 1.3806504e-23
consts.h = 6.6260896e-34
consts.m_e = 9.10938215e-31
consts.c = 2.99792458e8
consts.sigma_SB = 5.6704e-8
consts.e = 1.602176487e-19
consts.epsilon_0 = 8.854187817e-12
consts.G = 6.673e-11
consts.gamma = 5/3 # Assume non-relativistic adiabatic index to start
consts.M_sun = 1.9891e30
consts.R_sun = 6.95508e8
consts.L_sun = 3.84e26

def root_finder(f, x0, x1, epsilon=1e-6, **kwargs):
    """
    Finds root of a function in a particular range.
    f - function to find root of
    x0 - lower bound
    x1 - upper bound
    epsilon - error tolerance of root
    **kwargs - additional keyword arguments for f
    """

    f0 = f(x0, **kwargs)
    f1 = f(x1, **kwargs)

    if (np.sign(f0) == np.sign(f1)):
        fclosest = np.sign(f0)*min(np.abs(f0),np.abs(f1))
        if (fclosest == f0):
            return x0
        else:
            return x1

    xmid = (x0+x1)/2
    fmid = f(xmid, **kwargs)

    if (fmid == 0):
        return xmid
    else:
        if (np.sign(fmid) == np.sign(f1)):
            x1 = xmid
            f1 = fmid
        else:
            x0 = xmid
            f0 = fmid

    root_counter = 1
    while (abs(f1-f0) > epsilon):
        xmid = (x0+x1)/2
        fmid = f(xmid, **kwargs)

        if (fmid == 0):
            break
        else:
            if (np.sign(fmid) == np.sign(f1)):
                x1 = xmid
                f1 = fmid
            else:
                x0 = xmid
                f0 = fmid

        root_counter += 1

        if (root_counter == 1000):
            # May not converge due to lack of precision -- manual linear interpolation
            xmid = -f1 * (x1-x0)/(f1-f0) + x1
            break

    return xmid

def sigmoid_suppress(x, A, mu):
    """
    Sigmoid function of the form sigmoid(x) = 1/(1 + exp(A*(x-mu))).
    * x - independent variable
    * A - slope parameter
    * mu - shift parameter
    """
    return 1/(1 + np.exp(A*(x-mu)))

def clamp(val, low, high):
    """
    Returns value clamped between [low,high].
    * val - value to be clamped
    * low - lower bound
    * high - upper bound
    """
    return max(low, min(high, val))
