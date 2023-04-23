import math
import numpy as np
import time
import os, sys
from utils import DataContainer, consts, clamp
from structure import EOS
from derivatives import stellar_derivatives
from boundaries import estimate_boundary_vals
from integrate import rk4_integration

def single_stellar_integrator(STAR, perturbations, ODE_integrator, xi, max_steps):
    """
    Integrates equations of stellar structure for a star of a given mass.
    * STAR - Star object containing mass MSTAR and mass fractions X
    * perturbations - array of perturbations to [R,P_central,L_surf,T_central]
    * ODE_integrator - function to integrate PDEs of stellar structure
    * xi - multiplicative fraction for integration stepsize
    * max_steps - maximum number of iterations allowed for convergence
    """
    print("Beginning stellar integration pass...", flush=True)
    t1_single = time.time()

    boundaries = estimate_boundary_vals(STAR, perturbations)
    print("Central boundary conditions (r,P,L,T,rho):", boundaries[:5], flush=True)
    print("Surface boundary conditions (r,P,L,T,rho):", boundaries[5:], flush=True)
    assert (np.array(boundaries) >= 0).all()

    m1 = STAR.M * 1e-10
    z1_0 = np.array(boundaries[:4]) # [R_central, P_central, L_central, and T_central]
    rho_central = boundaries[4]
    m2 = STAR.M
    z2_0 = np.array(boundaries[5:-1]) # [R, P_surface, L_total, and T_effective]
    rho_surface = boundaries[-1]

    STAR.boundaries_central = boundaries[:4]
    STAR.rho_central = boundaries[4]
    STAR.boundaries_surface = boundaries[5:-1]
    STAR.rho_surface = boundaries[9]
    STAR.boundaries_min = [min(STAR.boundaries_central[i], STAR.boundaries_surface[i]) for i in range(len(STAR.boundaries_central))]
    STAR.boundaries_max = [max(STAR.boundaries_central[i], STAR.boundaries_surface[i]) for i in range(len(STAR.boundaries_central))]

    _, epsilon_central, kappa_central = EOS(z1_0, STAR)
    _, epsilon_surface, kappa_surface = EOS(z2_0, STAR)

    m1s = [m1]
    m2s = [m2]
    z1s = [z1_0]
    z2s = [z2_0]
    y1s = [np.array([rho_central, epsilon_central, kappa_central])]
    y2s = [np.array([rho_surface, epsilon_surface, kappa_surface])]
    z1 = np.copy(z1_0)
    z2 = np.copy(z2_0)

    meeting = False

    for i in range(1,max_steps):
        rk41 = False
        rk41_counter = 0
        rk42 = False
        rk42_counter = 0

        r1, P1, L1, T1 = z1
        r2, P2, L2, T2 = z2

        dz1dm = stellar_derivatives(z1, m1, STAR)
        dz2dm = stellar_derivatives(z2, m2, STAR)
        hz1 = np.abs(z1/dz1dm)
        hz2 = np.abs(z2/dz2dm)
        h1 = abs(min(m1*1e-2, xi*np.min(hz1)))
        h2 = -1*abs(min(m2*1e-2, xi*np.min(hz2)))
        if (h2 >= 0):
            print("Step:", i)
            print("h1:", h1)
            print("h2:", h2)
            print("m1:", m1)
            print("m2:", m2)
            raise Excpetion("h2 must be negative (not meeting)")
        if ((m1+h1) > (m2+h2)):
            m_mid = (m1+m2)/2
            h1 = m_mid-m1
            h2 = m_mid-m2
            meeting = True
            if (h2 >= 0):
                print("Step:", i)
                print("h1:", h1)
                print("h2:", h2)
                print("m1:", m1)
                print("m2:", m2)
                raise Exception("h2 must be negative (meeting)")

        while (rk41 == False):
            try:
                z1 = rk4_integration(stellar_derivatives, z1, m1, h1, STAR=STAR)
                z1 = np.array([clamp(z1[i], STAR.boundaries_min[i], STAR.boundaries_max[i]) for i in range(len(z1))])
                rk41 = True
                rho1, epsilon1, kappa1 = EOS(z1, STAR)
                y1 = np.array([rho1, epsilon1, kappa1])
            except:
                rk41_counter += 1
                h1 *= 0.1
                if (rk41_counter == 10):
                    rho1, epsilon1, kappa1 = EOS(z1, STAR)
                    rho2, epsilon2, kappa2 = EOS(z2, STAR)
                    print("Step:", i)
                    print("h1:", h1)
                    print("h2:", h2)
                    print("m1:", m1)
                    print("m2:", m2)
                    print("z1:", z1)
                    print("z2:", z2)
                    print("rho1:", rho1)
                    print("rho2:", rho2)
                    print("epsilon1:", epsilon1)
                    print("epsilon2:", epsilon2)
                    print("kappa1:", kappa1)
                    print("kappa2:", kappa2)
                    raise Exception("Outward RK4 integration failing!")
        while (rk42 == False):
            try:
                z2 = rk4_integration(stellar_derivatives, z2, m2, h2, STAR=STAR)
                z2 = np.array([clamp(z2[i], STAR.boundaries_min[i], STAR.boundaries_max[i]) for i in range(len(z2))])
                rk42 = True
                rho2, epsilon2, kappa2 = EOS(z2, STAR)
                y2 = np.array([rho2, epsilon2, kappa2])
            except:
                rk42_counter += 1
                h2 *= 0.1
                if (rk42_counter == 10):
                    rho1, epsilon1, kappa1 = EOS(z1, STAR)
                    rho2, epsilon2, kappa2 = EOS(z2, STAR)
                    print("Step:", i)
                    print("h1:", h1)
                    print("h2:", h2)
                    print("m1:", m1)
                    print("m2:", m2)
                    print("z1:", z1)
                    print("z2:", z2)
                    print("rho1:", rho1)
                    print("rho2:", rho2)
                    print("epsilon1:", epsilon1)
                    print("epsilon2:", epsilon2)
                    print("kappa1:", kappa1)
                    print("kappa2:", kappa2)
                    raise Exception("Inward RK4 integration failing!")

        m1 += h1
        m2 += h2

        if math.isclose(m1, m2, rel_tol=1e-12):
            meeting = True

        if np.isinf(z1).any() or np.isnan(z1).any():
            print("Step:", i)
            print("m1:", m1)
            print("z1:", z1)
            print("Previous z1:", z1s[-1])
            print("dz1dm:", dz1dm)
            print("hz1:", hz1)
            print("h1:", h1)
            raise Exception("Variable cannot take NaN or inf value")
        if np.isinf(z2).any() or np.isnan(z2).any():
            print("Step:", i)
            print("m2:", m2)
            print("z2:", z2)
            print("Previous z2:", z2s[-1])
            print("dz2dm:", dz2dm)
            print("hz2:", hz2)
            print("h2:", h2)
            raise Exception("Variable cannot take NaN or inf value")

        m1s.append(m1)
        m2s.append(m2)
        z1s.append(z1)
        z2s.append(z2)
        y1s.append(y1)
        y2s.append(y2)

        if (i % 500 == 0):
            print("Completed %i steps..."%i, flush=True)

        if (meeting == True):
            print("Step:", i)
            print("m1:", m1)
            print("z1:", z1)
            print("m2:", m2)
            print("z2:", z2)
            t2_single = time.time()
            break
    else: # Else on for loop -- activates if for loop exits normally
        print("m1:", m1)
        print("z1:", z1)
        print("m2:", m2)
        print("z2:", z2)
        t2_single = time.time()
        print("Integration pass time: %.2f min"%((t2_single-t1_single)/60))
        raise Exception("Maximum number of iterations reached without convergence")

    print("Integration pass time: %.2f min"%((t2_single-t1_single)/60), flush=True)
    return m1s, z1s, y1s, m2s, z2s, y2s

def stellar_perturbations(STAR, residuals, perturbations, ODE_integrator, xi, max_steps):
    """
    Calculates perturbations to boundary conditions using residuals between inward and outward solutions.
    * STAR - Star object containing mass MSTAR and mass fractions X
    * residuals - residuals between outward and inward solutions at midpoint
    * perturbations - array of initial perturbations to [R,P_central,L_surf,T_central]
    * ODE_integrator - function to integrate PDEs of stellar structure
    * xi - multiplicative fraction for integration stepsize
    * max_steps - maximum number of iterations allowed for convergence
    """
    print("Calculating perturbations...")
    t1_pert = time.time()
    residuals_matrix = np.zeros((4,4))

    boundaries = estimate_boundary_vals(STAR, perturbations)
    bcs = np.array([boundaries[5], boundaries[1], boundaries[7], boundaries[3]])
    bc_strings = ["R","P_central","L_surface","T_central"]

    for i, pert in enumerate(perturbations):
        print("Perturbing %s..."%bc_strings[i], flush=True)
        new_perturbations = np.copy(perturbations)
        if (pert == 0):
            new_perturbations[i] += 1e-2 * bcs[i]
        else:
            new_perturbations[i] *= 1.01
        m1s_pert, z1s_pert, y1s_pert, m2s_pert, z2s_pert, y2s_pert = single_stellar_integrator(STAR, new_perturbations, ODE_integrator, xi, max_steps)
        residuals_pert = z2s_pert[-1] - z1s_pert[-1]
        residuals_matrix[:,i] = (residuals_pert-residuals)/(new_perturbations[i]-perturbations[i])

    print(residuals_matrix)
    try:
        updated_perturbations = np.linalg.solve(residuals_matrix, -1*residuals)
    except:
        updated_perturbations = np.linalg.lstsq(residuals_matrix, -1*residuals)

    try:
        assert (np.abs(updated_perturbations/bcs) < 1).all()
    except:
        updated_perturbations = np.array([clamp(frac_pert, -0.1, 0.1) for frac_pert in updated_perturbations/bcs])*bcs
    t2_pert = time.time()
    print("Residuals/perturbations time: %.2f min"%((t2_pert-t1_pert)/60), flush=True)

    return updated_perturbations

def full_stellar_integrator(MSTAR, X, ODE_integrator=rk4_integration, xi=0.1, max_steps=10000, delta=1e-3):
    """
    Fully integrates equations of stellar structure with solution matching in the center.
    * MSTAR - total stellar mass
    * X - dictionary of mass fractions ["H":X_H, "He":X_He, "Li":X_Li, ..., "Z":Z_other]
    * ODE_integrator - function to integrate PDFs of stellar structure
    * xi - multiplicative fraction for integration stepsize
    * max_steps - maximum number of iterations allowed for convergence
    * delta - error tolerance for residuals
    """
    outfolder = os.getcwd()
    outfolder += "/outputs/"
    outmass = np.round(float(MSTAR/consts.M_sun),1)
    outfolder += "/M" + str(outmass) + "/"
    if not os.path.isdir(outfolder):
        os.mkdir(outfolder)

    print("Beginning stellar integration...", flush=True)
    t1_full = time.time()

    assert np.sum(list(X.values())) <= 1
    if "Z" not in X.keys():
        X["Z"] = 1-np.sum(list(X.values()))

    STAR = DataContainer()
    STAR.M = MSTAR
    STAR.X = X

    perturbations = np.zeros(4)

    m1s, z1s, y1s, m2s, z2s, y2s = single_stellar_integrator(STAR, perturbations, ODE_integrator, xi, max_steps)
    data = np.array([], dtype=[("m","f8"), ("r","f8"), ("L","f8"), ("P","f8"), ("T","f8"), ("rho","f8"), ("epsilon","O"), ("kappa","f8")])
    for i in range(len(m1s)):
        data = np.append(data, np.array([(m1s[i], z1s[i][0], z1s[i][1], z1s[i][2], z1s[i][3], y1s[i][0], y1s[i][1], y1s[i][2])], dtype=data.dtype))
    for i in reversed(range(len(m2s))):
        data = np.append(data, np.array([(m2s[i], z2s[i][0], z2s[i][1], z2s[i][2], z2s[i][3], y2s[i][0], y2s[i][1], y2s[i][2])], dtype=data.dtype))
    np.save(outfolder + "pass00.npy",data)
    
    residuals = z2s[-1] - z1s[-1]
    print("Residuals:", residuals, flush=True)

    new_perturbations = stellar_perturbations(STAR, residuals, perturbations, ODE_integrator, xi, max_steps)
    print("Perturbations:", new_perturbations, flush=True)

    orig_boundaries = estimate_boundary_vals(STAR, perturbations)
    orig_bcs = np.array([orig_boundaries[5], orig_boundaries[1], orig_boundaries[7], orig_boundaries[3]])

    residuals_counter = 1
    while (np.abs(residuals/orig_bcs) > delta).any():
        print("Fractional residuals:", np.abs(residuals/orig_bcs), flush=True)
        print("Residuals counter:", residuals_counter, flush=True)
        perturbations = np.copy(new_perturbations)

        m1s, z1s, y1s, m2s, z2s, y2s = single_stellar_integrator(STAR, perturbations, ODE_integrator, xi, max_steps)
        data = np.array([], dtype=[("m","f8"), ("r","f8"), ("L","f8"), ("P","f8"), ("T","f8"), ("rho","f8"), ("epsilon","O"), ("kappa","f8")])
        for i in range(len(m1s)):
            data = np.append(data, np.array([(m1s[i], z1s[i][0], z1s[i][1], z1s[i][2], z1s[i][3], y1s[i][0], y1s[i][1], y1s[i][2])], dtype=data.dtype))
        for i in reversed(range(len(m2s))):
            data = np.append(data, np.array([(m2s[i], z2s[i][0], z2s[i][1], z2s[i][2], z2s[i][3], y2s[i][0], y2s[i][1], y2s[i][2])], dtype=data.dtype))
        if (residuals_counter < 10):
            np.save(outfolder + "pass0" + str(residuals_counter) + ".npy",data)
        elif (residuals_counter < 100):
            np.save(outfolder + "pass" + str(residuals_counter) + ".npy",data)
        else:
            raise Exception("Too many perturbations without convergence")
        
        residuals = z2s[-1] - z1s[-1]
        print("Residuals:", residuals, flush=True)
        new_perturbations = stellar_perturbations(STAR, residuals, perturbations, ODE_integrator, xi, max_steps)

        residuals_counter += 1

    m1s, z1s, y1s, m2s, z2s, y2s = single_stellar_integrator(STAR, perturbations, ODE_integrator, xi, max_steps)
    data = np.array([], dtype=[("m","f8"), ("r","f8"), ("L","f8"), ("P","f8"), ("T","f8"), ("rho","f8"), ("epsilon","O"), ("kappa","f8")])
    for i in range(len(m1s)):
        data = np.append(data, np.array([(m1s[i], z1s[i][0], z1s[i][1], z1s[i][2], z1s[i][3], y1s[i][0], y1s[i][1], y1s[i][2])], dtype=data.dtype))
    for i in reversed(range(len(m2s))):
        data = np.append(data, np.array([(m2s[i], z2s[i][0], z2s[i][1], z2s[i][2], z2s[i][3], y2s[i][0], y2s[i][1], y2s[i][2])], dtype=data.dtype))
    if (residuals_counter < 10):
        np.save(outfolder + "pass0" + str(residuals_counter) + ".npy",data)
    elif (residuals_counter < 100):
        np.save(outfolder + "pass" + str(residuals_counter) + ".npy",data)
    else:
        raise Exception("Too many perturbations without convergence")

    r1s = np.array([zed[0] for zed in z1s])
    P1s = np.array([zed[1] for zed in z1s])
    L1s = np.array([zed[2] for zed in z1s])
    T1s = np.array([zed[3] for zed in z1s])

    rho1s = np.array([y[0] for y in y1s])
    epsilon1s = np.array([y[1] for y in y1s])
    kappa1s = np.array([y[2] for y in y1s])

    r2s = np.array([zed[0] for zed in z2s])
    P2s = np.array([zed[1] for zed in z2s])
    L2s = np.array([zed[2] for zed in z2s])
    T2s = np.array([zed[3] for zed in z2s])

    rho2s = np.array([y[0] for y in y2s])
    epsilon2s = np.array([y[1] for y in y2s])
    kappa2s = np.array([y[2] for y in y2s])

    #ms = np.concatenate((m1s[:-1],np.array([np.mean(m1s[-1],m2s[-1])]),m2s[::-1][1:]))
    #rs = np.concatenate((r1s[:-1],np.array([np.mean(r1s[-1],r2s[-1])]),r2s[::-1][1:]))
    #Ps = np.concatenate((P1s[:-1],np.array([np.mean(P1s[-1],P2s[-1])]),P2s[::-1][1:]))
    #Ls = np.concatenate((L1s[:-1],np.array([np.mean(L1s[-1],L2s[-1])]),L2s[::-1][1:]))
    #Ts = np.concatenate((T1s[:-1],np.array([np.mean(T1s[-1],T2s[-1])]),T2s[::-1][1:]))
    t2_full = time.time()
    print("Integration complete!", flush=True)
    print("Full integration time: %.2f min"%((t2_full-t1_full)/60), flush=True)

    #return ms, rs, Ps, Ls, Ts
    return m1s, r1s, P1s, L1s, T1s, rho1s, epsilon1s, kappa1s, m2s, r2s, P2s, L2s, T2s, rho2s, epsilon2s, kappa2s
