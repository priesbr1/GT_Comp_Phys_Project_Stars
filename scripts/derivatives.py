import math
import numpy as np
from utils import consts
from structure import EOS, temperature_gradient

def stellar_derivatives(z, m, STAR):
    """
    Calculates stellar derivatives (dz/dm).
    * z - array of current values of [r, P, L, T]
    * m - current mass value
    * STAR - Star object containing mass MSTAR, mass fractions X, and central/surface boundary conditions
    """
    r, P, L, T = z

    rho, epsilon, kappa = EOS(z, STAR)

    drdm = 1/(4*np.pi * r**2 * rho)
    dPdm = -1/(4*np.pi * r**2) * (consts.G*m/r**2)
    dLdm = np.sum(list(epsilon.values()))
    dTdm = temperature_gradient(STAR.X, z, rho, kappa, dPdm)

    dzdm = np.array([drdm, dPdm, dLdm, dTdm])

    return dzdm
