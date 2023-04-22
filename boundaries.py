import math
import numpy as np
from utils import consts, sigmoid_suppress
from structure import molecular_weights, opacity_free_free, opacity_bound_free, opacity_Hminus, reduced_luminosity

def estimate_boundary_vals(STAR, perturbations):
    """
    Estimates values of integration variables at stellar center and surface
    * STAR - Star object containing mass MSTAR and mass fractions X
    * perturbations - array of perturbations to [R,P_central,L_surf,T_central]
    """
    mu_ions, mu_e = molecular_weights(STAR.X)

    # Surface values
    L_surface = (STAR.M/consts.M_sun)**3.5 * consts.L_sun + perturbations[2]
    if (STAR.M <= consts.M_sun):
        R = (STAR.M/consts.M_sun)**0.8 * consts.R_sun + perturbations[0]
    else:
        R = (STAR.M/consts.M_sun)**0.57 * consts.R_sun + perturbations[0]
    T_surface = (L_surface/(4*np.pi*consts.sigma_SB * R**2))**(1/4)
    rho_surface = np.sqrt(2/3)/np.sqrt(3.68e22 * ((1-STAR.X["H"]-STAR.X["He"])**2)/(mu_ions*mu_e) * T_surface**(-7/2) + 4.34e25 * (1-STAR.X["H"]-STAR.X["He"]) * (1+STAR.X["H"]) * T_surface**(-7/2) +
                                       (2.5e-32 * (1-STAR.X["H"]-STAR.X["He"])/0.02 * T_surface**9 * sigmoid_suppress(T_surface, 2, 5000))**2)
    kappa_surface = opacity_free_free(STAR.X, rho_surface, T_surface) + opacity_bound_free(STAR.X, rho_surface, T_surface) + opacity_Hminus(STAR.X, rho_surface, T_surface)
    P_surface = 2/3 * (consts.G*STAR.M)/(kappa_surface*R**2) + 2/(3*consts.c) * L_surface/(4*np.pi*R**2)

    # Central values
    rho_central = 115.0 * 3/(4*np.pi) * (STAR.M/R**3)
    R_central = (3/(4*np.pi) * (STAR.M*1e-10)/rho_central)**(1/3)
    P_central = 7.701*consts.G * (STAR.M**2)/R**4 + perturbations[1]
    T_central = 2/3 * consts.G/consts.k_B * STAR.M*consts.m_u/R + perturbations[3]
    epsilon = reduced_luminosity(STAR.X, rho_central, T_central)
    L_central = np.sum(list(epsilon.values())) * STAR.M*1e-10

    return R_central, P_central, L_central, T_central, rho_central, R, P_surface, L_surface, T_surface, rho_surface
