"""
Preliminary structural sizing of the leg links as hollow circular tubes (FDM PLA / PETG).

Wall thickness t is fixed (MIN_WALL_THICKNESS_MM); the unknown is the outer radius R.
    A = pi (R^2 - Ri^2),  I = pi (R^4 - Ri^4) / 4,  J = 2 I,  Ri = max(R - t, 0)
    sigma_adm = sigma_yield * ANISOTROPY_FACTOR / FS_MATERIAL

1. Axial criterion (as specified):  sigma = |N|max / A <= sigma_adm
     A_min = |N|max / sigma_adm  ->  R = (A_min / pi + t^2) / (2 t)
   If that R is below t the tube is wall-limited: the smallest part is a solid rod of radius t.
2. Combined criterion (axial + bending + shear + torsion, von Mises), evaluated with the load
   components acting SIMULTANEOUSLY at every time sample and at both ends of the link:
     sigma = |N|/A + Mb R / I ,   tau = 2 |V| / A + |T| R / J ,   sigma_vm = sqrt(sigma^2 + 3 tau^2)
   (2V/A is the peak shear of a thin-walled tube; conservative for thicker walls).
   The smallest R with max_t sigma_vm <= sigma_adm is found by bisection.
"""
import numpy as np

from . import config as C
from .model_builder import LINK_NAMES, tube_area


def _section(R, t):
    Ri = np.maximum(R - t, 0.0)
    A = np.pi * (R ** 2 - Ri ** 2)
    I = np.pi * (R ** 4 - Ri ** 4) / 4.0
    return A, I, 2.0 * I


def von_mises(R, t, N, V, Mb, T):
    A, I, J = _section(R, t)
    sig = np.abs(N) / A + Mb * R / I
    tau = 2.0 * V / A + np.abs(T) * R / J
    return np.sqrt(sig ** 2 + 3.0 * tau ** 2), np.abs(N) / A, Mb * R / I, tau


def sigma_adm(material):
    return C.MATERIALS[material]["sigma_yield_MPa"] * 1e6 * C.ANISOTROPY_FACTOR / C.FS_MATERIAL


def size_link(N, V, Mb, T, material, length):
    """N, V, Mb, T: (S, 2) load components of one link. Returns the sizing dict (SI units)."""
    t = C.MIN_WALL_THICKNESS_MM / 1000.0
    r_min, r_max = C.MIN_OUTER_RADIUS_MM / 1000.0, C.MAX_OUTER_RADIUS_MM / 1000.0
    sa = sigma_adm(material)

    # 1. axial only
    N_max = float(np.abs(N).max())
    A_min = N_max / sa
    R_ax_formula = (A_min / np.pi + t ** 2) / (2 * t)
    R_ax = max(R_ax_formula, r_min)

    # 2. combined (von Mises)
    worst = lambda R: float(von_mises(R, t, N, V, Mb, T)[0].max())
    if worst(r_min) <= sa:
        R_c = r_min
    elif worst(r_max) > sa:
        R_c = np.inf
    else:
        lo, hi = r_min, r_max
        for _ in range(60):
            mid = 0.5 * (lo + hi)
            lo, hi = (lo, mid) if worst(mid) <= sa else (mid, hi)
        R_c = hi
    governing = "-"
    if np.isfinite(R_c):
        vm, s_ax, s_b, tau = von_mises(R_c, t, N, V, Mb, T)
        i = np.unravel_index(np.argmax(vm), vm.shape)
        parts = {"axial": s_ax[i], "bending": s_b[i], "shear+torsion": np.sqrt(3) * tau[i]}
        governing = max(parts, key=parts.get)
    rho = C.MATERIALS[material]["density_kg_m3"]
    return dict(
        N_max=N_max, A_min_axial=A_min, R_axial_formula=R_ax_formula, R_axial=R_ax,
        wall_limited_axial=R_ax_formula < r_min,
        R_combined=R_c, A_combined=tube_area(R_c, t) if np.isfinite(R_c) else np.inf,
        governing=governing, mass=rho * tube_area(R_c, t) * length if np.isfinite(R_c) else np.inf,
        sigma_adm=sa, length=length,
    )


def size_all(loads, material, lengths):
    return {link: size_link(loads.N[:, i], loads.V[:, i], loads.Mb[:, i], loads.T[:, i], material, lengths[link])
            for i, link in enumerate(LINK_NAMES)}
