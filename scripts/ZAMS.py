from utils import consts
from stellar import full_stellar_integrator

M = consts.M_sun
X = {
    "H": 0.7346, "He": 0.2485, "C": 0.0029, "N": 0.0009, "O": 0.0077, "Ne": 0.0012, "Mg": 0.0005, "Si": 0.0007, "S": 0.0004, "Fe": 0.0016
}

#ms, rs, Ps, Ls, Ts = full_stellar_integrator(M, X)
m1s, r1s, P1s, L1s, T1s, rho1s, epsilon1s, kappa1s, m2s, r2s, P2s, L2s, T2s, rho2s, epsilon2s, kappa2s = full_stellar_integrator(M, X)
