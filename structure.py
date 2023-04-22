import math
import numpy as np
from utils import consts, root_finder, sigmoid_suppress

def molecular_weights(X):
    """
    Calculates mean molecular weights for ions (mu_ions) and electrons (mu_e).
    * X - dictionary of mass fractions ["H":X_H, "He":X_He, "Li":X_Li, ..., "Z":Z_other]
    """
    mu_ions = 1/np.sum([X[species]/consts.A[species] for species in X.keys()])
    mu_e = 1/np.sum([X[species]*consts.Z[species]/consts.A[species] for species in X.keys()])

    return mu_ions, mu_e

def pressure_ions(X, rho, T):
    """
    Calculates local ion pressure (P_ions).
    * X - dictionary of mass fractions ["H":X_H, "He":X_He, "Li":X_Li, ..., "Z":Z_other]
    * rho - local density
    * T - local temperature
    """
    mu_ions, mu_e = molecular_weights(X)

    P_ions = rho/(mu_ions*consts.m_u) * consts.k_B*T

    return P_ions

def pressure_electrons(X, rho, T):
    """
    Calculates local electron pressure (P_e).
    * X - dictionary of mass fractions ["H":X_H, "He":X_He, "Li":X_Li, ..., "Z":Z_other]
    * rho - local density
    * T - local temperature
    """
    mu_ions, mu_e = molecular_weights(X)

    rho_trans_degenerate = (20*consts.m_e*consts.k_B)**(3/2) * (np.pi*consts.m_u/(3*consts.h**3)) * mu_e * T**(3/2)
    rho_trans_relativistic = (5/2 * consts.m_e*consts.c/consts.h)**3 * (np.pi*consts.m_u/3) * mu_e

    if (rho < rho_trans_degenerate):
        P_e = rho/(mu_e*consts.m_u) * consts.k_B*T
        consts.gamma = 5/3 # Non-relativistic adiabatic index
    elif (rho < rho_trans_relativistic):
        P_e = consts.h**2/(20*consts.m_e) * (3/np.pi)**(2/3) * (rho/(mu_e*consts.m_u))**(5/3)
        consts.gamma = 5/3 # Non-relativistic adiabatic index
    else:
        P_e = consts.h*consts.c/8 * (3/np.pi)*(1/3) * (rho/(mu_e*consts.m_u))**(4/3)
        consts.gamma = 4/3 # Relativistic adiabatic index

    return P_e

def pressure_gas(X, rho, T):
    """
    Calculates local gas pressure (P_gas).
    * X - dictionary of mass fractions ["H":X_H, "He":X_He, "Li":X_Li, ..., "Z":Z_other]
    * rho - local density
    * T - local temperature
    """
    P_ions = pressure_ions(X, rho, T)
    P_e = pressure_electrons(X, rho, T)

    return P_ions + P_e

def pressure_radiation(T):
    """
    Calculates local radiation pressure (P_rad).
    * T - local temperature
    """
    P_rad = 4/3 * consts.sigma_SB/consts.c * T**4

    return P_rad

def pressure(X, rho, T):
    """
    Calculates local pressure (P).
    * X - dictionary of mass fractions ["H":X_H, "He":X_He, "Li":X_Li, ..., "Z":Z_other]
    * rho - local pressure
    * T - local temperature
    """
    P_ions = pressure_ions(X, rho, T)
    P_e = pressure_electrons(X, rho, T)
    P_rad = pressure_radiation(T)

    return P_ions + P_e + P_rad

def pressure_gas_difference(rho, X, P, T):
    """
    Calculates difference between P_gas and pressure_gas(X, rho, T) for density root finding.
    * rho - local density
    * X - dictionary of mass fractions ["H":X_H, "He":X_He, "Li":X_Li, ..., "Z":Z_other]
    * P - local pressure
    * T - local temperature
    """
    P_gas = P - pressure_radiation(T)

    return P_gas - pressure_gas(X, rho, T)

def pressure_difference(rho, X, P, T):
    """
    Calculates difference between P and pressure(X, rho, T) for density root finding.
    * rho - local density
    * X - dictionary of mass fractions ["H":X_H, "He":X_He, "Li:"X_Li, ..., "Z":Z_other]
    * P - local pressure
    * T - local temperature
    """
    return P - pressure(X, rho, T)

def density(X, P, T, rho_surface, rho_central):
    """
    Calculates local density (rho).
    * X - dictionary of mass fractions ["H":X_H, "He":X_He, "Li:"X_Li, ..., "Z":Z_other]
    * P - local pressure
    * T - local temperature
    * rho_surface - surface density
    * rho_central - central density
    """
    rho = root_finder(pressure_difference, rho_surface, rho_central, X=X, P=P, T=T)

    return rho

def reduced_luminosity(X, rho, T):
    """
    Calculates local reduced luminosity (epsilon).
    * X - dictionary of mass fractions ["H":X_H, "He":X_He, "Li":X_Li, ..., "Z":Z_other]
    * rho - local density
    * T - local temperature
    """
    epsilon = dict()

    if all([species in X.keys() for species in ["H"]]):
        epsilon_pp = 2.4e3 * (rho * X["H"]**2 / T**(2/3)) * np.exp(-3.380e3 / T**(1/3))
        epsilon[r"$pp$-chain"] = epsilon_pp
    if all([species in X.keys() for species in ["H","C","N","O"]]):
        epsilon_CNO = 4.4e24 * (rho*X["H"]*(X["C"]+X["N"]+X["O"]) / T**(2/3)) * np.exp(-15.288e3 / T**(1/3))
        epsilon[r"$\mathrm{CNO}"] = epsilon_CNO
    if all([species in X.keys() for species in ["He"]]):
        epsilon_3alpha = 5.1e28 * (rho**2 * X["He"]**3 / T**3) * np.exp(-4.4027e9 / T)
        epsilon[r"3-$\alpha$"] = epsilon_3alpha
    # Additional energy generation rates for higher-order burning are located in Penn State Lec. 22

    return epsilon

def opacity_electrons_radiative(X):
    """
    Calculates local opacity from electron scattering (kappa_e_rad).
    * X - dictionary of mass fractions ["H":X_H, "He":X_He, "Li":X_Li, ..., "Z":Z_other]
    """
    kappa_e_rad = 0.02*(1+X["H"])

    return kappa_e_rad

def opacity_free_free(X, rho, T):
    """
    Calculates local opacity from free-free absorption (kappa_ff).
    * X - dictionary of mass fractions ["H":X_H, "He":X_He, "Li":X_Li, ..., "Z":Z_other]
    * rho - local density
    * T - local temperature
    """
    mu_ions, mu_e = molecular_weights(X)

    kappa_ff = 3.68e22 * ((1-X["H"]-X["He"])**2)/(mu_ions*mu_e) * rho * T**(-7/2)

    return kappa_ff

def opacity_bound_free(X, rho, T):
    """
    Calculates local opacity from bound-free absorption (kappa_bf).
    * X - dictionary of mass fractions ["H":X_H, "He":X_He, "Li":X_Li, ..., "Z":Z_other]
    * rho - local density
    * T - local temperature
    """
    guillotine = 1/(2.82*(rho*(1+X["H"]))**0.2)
    kappa_bf = 4.34e25 * guillotine * (1-X["H"]-X["He"])*(1+X["H"]) * rho * T**(-7/2)

    return kappa_bf

def opacity_Hminus(X, rho, T):
    """
    Calculates local opacity from H- absorption (kappa_Hminus).
    Includes artificial suppression to account for ionization fraction.
    * X - dictionary of mass fractions ["H":X_H, "He":X_He, "Li":X_Li, ..., "Z":Z_other]
    * rho - local density
    * T - local temperature
    """
    kappa_Hminus = 2.5e-32 * (1-X["H"]-X["He"])/0.02 * np.sqrt(rho) * T**9 * sigmoid_suppress(T, 2, 5000)

    return kappa_Hminus

def opacity_radiative(X, rho, T):
    """
    Calculates local radiative opacity (kappa_rad).
    * X - dictionary of mass fractions ["H":X_H, "He":X_He, "Li":X_Li, ..., "Z":Z_other]
    * rho - local density
    * T - local temperature
    """
    kappa_e_rad = opacity_electrons_radiative(X)
    kappa_ff = opacity_free_free(X, rho, T)
    kappa_bf = opacity_bound_free(X, rho, T)
    kappa_Hminus = opacity_Hminus(X, rho, T)

    return kappa_e_rad + kappa_ff + kappa_bf + kappa_Hminus

def opacity_electrons_conductive(X, rho, T):
    """
    Calculates local opacity from electron conduction (kappa_e_cond).
    * X - dictionary of mass fractions ["H":X_H, "He":X_He, "Li":X_Li, ..., "Z":Z_other]
    * rho - local density
    * T - local temperature
    """
    Zbar = np.sum([X[species]*consts.Z[species] for species in X.keys()])

    kappa_e_cond = (32*np.pi*consts.sigma_SB * T**3)/(3*consts.k_B * rho) * np.sqrt(consts.m_e/(3*consts.k_B*T)) \
                   * ((Zbar*consts.e**2)/(4*np.pi*consts.epsilon_0*consts.k_B*T))**2

    return kappa_e_cond

def opacity_photons_conductive(X, rho, T):
    """
    Calculates local opacity from photon conduction (kappa_gamma).
    * X - dictionary of mass fractions ["H":X_H, "He":X_He, "Li":X_Li, ..., "Z":Z_other]
    * rho - local density
    * T - local temperature
    """
    Zbar = np.sum([X[species]*consts.Z[species] for species in X.keys()])
    P_rad = pressure_radiation(T)
    P_e = pressure_electrons(X, rho, T)

    kappa_e_cond = opacity_electrons_conductive(X, rho, T)
    kappa_gamma = kappa_e_cond * np.sqrt(3) * Zbar * P_rad/P_e * ((consts.m_e * consts.c**2)/(consts.k_B*T))**(5/2)

    return kappa_gamma

def opacity_conductive(X, rho, T):
    """
    Calculates local conductive opacity (kappa_cond).
    * X - dictionary of mass fractions ["H":X_H, "He":X_He, "Li":X_Li, ..., "Z":Z_other]
    * rho - local density
    * T - local temperature
    """
    kappa_e_cond = opacity_electrons_conductive(X, rho, T)
    kappa_gamma = opacity_photons_conductive(X, rho, T)

    return kappa_e_cond + kappa_gamma

def opacity(X, rho, T):
    """
    Calculates local opacity (kappa).
    * X - dictionary of mass fractions ["H":X_H, "He":X_He, "Li":X_Li, ..., "Z":Z_other]
    * rho - local density
    * T - local temperature
    """
    kappa_rad = opacity_radiative(X, rho, T)
    kappa_cond = opacity_conductive(X, rho, T)

    return 1/(1/kappa_rad + 1/kappa_cond)

def temperature_gradient(X, z, rho, kappa, dPdm):
    """
    Calculates temperature gradient based on dominant energy transfer mechanism.
    * X - dictionary of mass fractions ["H":X_H, "He":X_He, "Li":X_Li, ..., "Z":Z_other]
    * z - array of [r,P,L,T]
    * rho - local density
    * kappa - local opacity
    * dPdm - local pressure gradient
    """
    r, P, L, T = z

    mu_ions, mu_e = molecular_weights(X)
    rho_trans_degenerate = (20*consts.m_e*consts.k_B)**(3/2) * (np.pi*consts.m_u/(3*consts.h**3)) * mu_e * T**(3/2)

    if (rho < rho_trans_degenerate):
        dPdT = rho*consts.k_B/consts.m_u * (1/mu_ions + 1/mu_e) + 16/3 * consts.sigma_SB/consts.c * T**3
    else:
        dPdT = rho*consts.k_B/(mu_ions*consts.m_u) + 16/3 * consts.sigma_SB/consts.c * T**3

    if (P/T / dPdT) < (1-1/consts.gamma): # radiative/conductive
        dTdm = -3*kappa*L/(256*np.pi**2 * r**4 * consts.sigma_SB * T**3)
    else:
        dTdm = (1-1/consts.gamma) * T/P * dPdm

    return dTdm

def EOS(z, STAR):
    """
    Function that calculates EOS variables (density, reduced luminosity, opacity).
    * z - array of current values of [r, P, L, T]
    * STAR - Star object containing mass MSTAR, mass fractions X, and central/surface boundary conditions
    """
    r, P, L, T = z

    rho = density(STAR.X, P, T, STAR.rho_surface, STAR.rho_central)
    epsilon = reduced_luminosity(STAR.X, rho, T)
    kappa_rad = opacity_radiative(STAR.X, rho, T)
    kappa_cond = opacity_conductive(STAR.X, rho, T)
    kappa = opacity(STAR.X, rho, T)

    return rho, epsilon, kappa
