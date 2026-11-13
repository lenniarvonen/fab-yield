"""Textbook process and device physics for an NMOS transistor.

Everything in this module is deterministic and vectorised over dies: every
function accepts numpy arrays (one entry per die) and returns arrays.

Process flow modelled (see docs/process_flow.md):
    1. Vt-adjust boron implant (dose, energy)     -> Gaussian as-implanted profile
    2. Activation anneal (temperature, time)       -> Arrhenius diffusion
    3. Dry gate oxidation (temperature, time)      -> Deal-Grove oxide growth,
                                                      extra diffusion, consumed Si
    4. n+ poly gate patterned at gate length L
    5. Electrical test: Vt, Ion, Ioff

References (also cited in README):
    [DG]   B. E. Deal, A. S. Grove, J. Appl. Phys. 36, 3770 (1965).
    [PDG]  J. D. Plummer, M. D. Deal, P. B. Griffin, "Silicon VLSI Technology",
           Prentice Hall (2000): Deal-Grove constants (Table 6-2), boron
           diffusivity (Table 7-5), implant range tables (ch. 8).
    [GIB]  J. F. Gibbons, W. S. Johnson, S. W. Mylroie, "Projected Range
           Statistics", 2nd ed. (1975): boron Rp / dRp in Si.
    [SZE]  S. M. Sze, K. K. Ng, "Physics of Semiconductor Devices", 3rd ed.
    [TSI]  Y. Tsividis, "Operation and Modeling of the MOS Transistor".
    [HU]   C. Hu, "Modern Semiconductor Devices for Integrated Circuits" (2010):
           velocity-saturation drain current, subthreshold swing.
    [LIU]  Z.-H. Liu et al., IEEE TED 40, 86 (1993): Vt roll-off / DIBL model.
    [CT]   D. M. Caughey, R. E. Thomas, Proc. IEEE 55, 2192 (1967): mobility.
"""

from __future__ import annotations

import numpy as np

# --------------------------------------------------------------------------
# Physical constants (CGS-ish units used throughout: cm, s, V, A, F)
# --------------------------------------------------------------------------
Q = 1.602176634e-19  # C
K_B_EV = 8.617333262e-5  # eV/K
EPS0 = 8.8541878128e-14  # F/cm
EPS_SI = 11.7 * EPS0
EPS_OX = 3.9 * EPS0
NI = 1.0e10  # cm^-3 intrinsic carrier conc. at 300 K
T_DEVICE = 300.0  # K, electrical test temperature
VT_THERMAL = K_B_EV * T_DEVICE  # kT/q in V (~25.85 mV)
EG_HALF = 0.56  # half the Si band gap, eV

# --------------------------------------------------------------------------
# Fixed technology parameters (documented to the agent in docs/device.md)
# --------------------------------------------------------------------------
N_SUB = 1.0e16  # p-type substrate boron concentration, cm^-3
WIDTH_CM = 10.0e-4  # transistor width W = 10 um
VDD = 3.3  # supply voltage, V (Vgs = Vds = Vdd for Ion)
VBI = 0.9  # source/drain-to-channel built-in potential, V (roll-off model)
QF_NOMINAL = 5.0e10  # fixed oxide charge, q/cm^2
NATIVE_OXIDE_CM = 1.5e-7  # 1.5 nm native oxide before gate oxidation
SI_CONSUMED_PER_OX = 0.44  # thickness of Si consumed per unit oxide grown
VT_DEFINITION_CURRENT = 1.0e-7  # A * (W/L): constant-current Vt definition
MU_N0 = 450.0  # low-field electron inversion-layer mobility, cm^2/Vs
THETA = 0.12  # mobility degradation coefficient, 1/V (scaled by 10nm/tox)
VSAT = 8.0e6  # electron saturation velocity, cm/s

# --------------------------------------------------------------------------
# 1. Implantation: boron range statistics in Si [GIB], [PDG ch.8]
# --------------------------------------------------------------------------
_IMPL_E_KEV = np.array([5.0, 10.0, 20.0, 30.0, 40.0, 60.0, 80.0, 100.0])
_IMPL_RP_NM = np.array([17.8, 33.3, 66.2, 98.7, 130.2, 190.3, 246.5, 298.8])
_IMPL_DRP_NM = np.array([10.5, 17.1, 28.3, 37.3, 44.3, 55.6, 64.1, 71.0])


def implant_range(energy_kev):
    """Projected range Rp and straggle dRp (cm) of boron in Si."""
    e = np.asarray(energy_kev, dtype=float)
    rp = np.interp(e, _IMPL_E_KEV, _IMPL_RP_NM) * 1e-7
    drp = np.interp(e, _IMPL_E_KEV, _IMPL_DRP_NM) * 1e-7
    return rp, drp


# --------------------------------------------------------------------------
# 2. Diffusion: intrinsic boron diffusivity, Arrhenius [PDG Table 7-5]
# --------------------------------------------------------------------------
B_D0 = 0.76  # cm^2/s
B_EA = 3.46  # eV


def boron_diffusivity(temp_c):
    """Intrinsic boron diffusivity D = D0 exp(-Ea/kT) in cm^2/s."""
    t_k = np.asarray(temp_c, dtype=float) + 273.15
    return B_D0 * np.exp(-B_EA / (K_B_EV * t_k))


# --------------------------------------------------------------------------
# 3. Oxidation: Deal-Grove, dry O2 on (100) Si [DG], [PDG Table 6-2]
# --------------------------------------------------------------------------
DG_B_C = 7.72e2  # um^2/h, parabolic rate prefactor (dry)
DG_B_EA = 1.23  # eV
DG_BA_C = 6.23e6 / 1.68  # um/h, linear rate prefactor (dry, (100) = (111)/1.68)
DG_BA_EA = 2.00  # eV


def deal_grove_rates(temp_c):
    """Return (B [um^2/h], B/A [um/h]) for dry oxidation of (100) Si."""
    t_k = np.asarray(temp_c, dtype=float) + 273.15
    kt = K_B_EV * t_k
    b = DG_B_C * np.exp(-DG_B_EA / kt)
    b_over_a = DG_BA_C * np.exp(-DG_BA_EA / kt)
    return b, b_over_a


def oxide_thickness(temp_c, time_min, x_initial_cm=NATIVE_OXIDE_CM):
    """Deal-Grove oxide thickness (cm) after `time_min` of dry oxidation.

    x^2 + A x = B (t + tau),  tau = (x_i^2 + A x_i) / B
    """
    b, b_over_a = deal_grove_rates(temp_c)
    a = b / b_over_a  # um
    xi = x_initial_cm * 1e4  # um
    tau = (xi**2 + a * xi) / b  # h
    t_h = np.asarray(time_min, dtype=float) / 60.0
    x_um = 0.5 * a * (np.sqrt(1.0 + 4.0 * b * (t_h + tau) / a**2) - 1.0)
    return x_um * 1e-4


# --------------------------------------------------------------------------
# Doping profile after the full thermal budget
# --------------------------------------------------------------------------
X_GRID_CM = np.linspace(0.0, 1.0e-4, 801)  # 0 .. 1 um below the Si/SiO2 interface


def channel_profile(dose, energy_kev, dt_total, si_consumed_cm, x=X_GRID_CM):
    """Net acceptor concentration N(x) (cm^-3) below the final Si/SiO2 interface.

    The as-implanted Gaussian (Rp, dRp) broadens by diffusion: sigma^2 =
    dRp^2 + 2 D t (a Gaussian stays Gaussian under constant-D diffusion).  A
    reflecting boundary at the original surface is modelled with an image
    source.  Oxidation consumes `si_consumed_cm` of silicon; dopant in the
    consumed layer is lost to the oxide (simplified segregation).

    Arrays: dose, energy, dt_total, si_consumed have shape (n,), output (n, len(x)).
    """
    dose = np.atleast_1d(np.asarray(dose, dtype=float))[:, None]
    rp, drp = implant_range(energy_kev)
    rp = np.atleast_1d(rp)[:, None]
    drp = np.atleast_1d(drp)[:, None]
    sig2 = drp**2 + 2.0 * np.atleast_1d(dt_total)[:, None]
    sig = np.sqrt(sig2)
    xs = x[None, :] + np.atleast_1d(si_consumed_cm)[:, None]  # depth in original Si coords
    norm = dose / (np.sqrt(2.0 * np.pi) * sig)
    n_impl = norm * (np.exp(-((xs - rp) ** 2) / (2 * sig2)) + np.exp(-((xs + rp) ** 2) / (2 * sig2)))
    # Gaussian + image integrates to exactly `dose` over x >= 0 (original coords).
    return N_SUB + n_impl


def thermal_budget(anneal_temp_c, anneal_time_min, ox_temp_c, ox_time_min):
    """Total D*t (cm^2) for boron from the anneal plus the gate oxidation."""
    return boron_diffusivity(anneal_temp_c) * np.asarray(anneal_time_min) * 60.0 + boron_diffusivity(
        ox_temp_c
    ) * np.asarray(ox_time_min) * 60.0


# --------------------------------------------------------------------------
# Device: depletion with a non-uniform profile [TSI ch.4], [SZE ch.6]
# --------------------------------------------------------------------------
def depletion(profile, x=X_GRID_CM):
    """Solve the depletion approximation at strong-inversion onset.

    Surface potential psi_s = 2 phi_F (phi_F from the substrate doping).
    Poisson in the depletion region: psi(W) = (q/eps) * int_0^W x N(x) dx.
    Returns (W_dmax [cm], Q_dep [C/cm^2], phi_F [V]).
    """
    phi_f = VT_THERMAL * np.log(N_SUB / NI)
    psi_target = 2.0 * phi_f
    dx = x[1] - x[0]
    moment = np.cumsum(profile * x[None, :], axis=1) * dx * Q / EPS_SI  # psi(W) for each W
    charge = np.cumsum(profile, axis=1) * dx * Q
    n = profile.shape[0]
    w = np.empty(n)
    qd = np.empty(n)
    for i in range(n):
        j = np.searchsorted(moment[i], psi_target)
        j = min(max(j, 1), len(x) - 1)
        # linear interpolation between grid points
        m0, m1 = moment[i, j - 1], moment[i, j]
        f = 0.0 if m1 == m0 else (psi_target - m0) / (m1 - m0)
        w[i] = x[j - 1] + f * dx
        qd[i] = charge[i, j - 1] + f * (charge[i, j] - charge[i, j - 1])
    return w, qd, phi_f


def device_characteristics(tox_cm, profile, gate_length_cm, qf=QF_NOMINAL):
    """Long-channel MOSFET + roll-off + velocity saturation.

    Returns dict of arrays: vt [V], ion [A], ioff [A], ss [V/dec],
    w_dep [cm], vt_long [V], dvt_rolloff [V].
    """
    tox_cm = np.asarray(tox_cm, dtype=float)
    lg = np.asarray(gate_length_cm, dtype=float)
    cox = EPS_OX / tox_cm
    w_dep, q_dep, phi_f = depletion(profile)

    # Flat band: n+ poly gate on p-type channel, phi_ms = -(Eg/2 + phi_F) [SZE]
    phi_ms = -(EG_HALF + phi_f)
    vfb = phi_ms - Q * np.asarray(qf) / cox
    vt_long = vfb + 2.0 * phi_f + q_dep / cox

    # Short-channel Vt roll-off and DIBL at Vds = Vdd [LIU]
    lt = np.sqrt(EPS_SI * tox_cm * w_dep / EPS_OX)  # characteristic length
    rolloff_factor = np.exp(-lg / (2.0 * lt)) + 2.0 * np.exp(-lg / lt)
    dvt = (2.0 * (VBI - 2.0 * phi_f) + VDD) * rolloff_factor
    vt = vt_long - dvt

    # Subthreshold swing [HU]: S = ln(10) kT/q (1 + Cdep/Cox)
    cdep = EPS_SI / w_dep
    ss = np.log(10.0) * VT_THERMAL * (1.0 + cdep / cox)
    ioff = VT_DEFINITION_CURRENT * (WIDTH_CM / lg) * 10.0 ** (-vt / ss)

    # On current, velocity-saturation model [HU eq. 6.x]
    vgt = np.maximum(VDD - vt, 1e-3)
    mu_eff = MU_N0 / (1.0 + THETA * (1.0e-6 / tox_cm) * vgt)
    ec = 2.0 * VSAT / mu_eff
    ion = WIDTH_CM * cox * VSAT * vgt**2 / (vgt + ec * lg)
    return {
        "vt": vt,
        "ion": ion,
        "ioff": ioff,
        "ss": ss,
        "w_dep": w_dep,
        "vt_long": vt_long,
        "dvt_rolloff": dvt,
    }


# --------------------------------------------------------------------------
# Sheet resistance of the boron layer (junction-isolated monitor wafer)
# --------------------------------------------------------------------------
def hole_mobility(n):
    """Caughey-Thomas hole mobility (cm^2/Vs) in Si at 300 K [CT]."""
    mu_min, mu_max, n_ref, alpha = 44.9, 470.5, 2.23e17, 0.719
    return mu_min + (mu_max - mu_min) / (1.0 + (n / n_ref) ** alpha)


def sheet_resistance(profile_impl_only, x=X_GRID_CM):
    """Rs = 1 / (q * int mu(N) N dx) in ohm/sq for the implanted layer only."""
    dx = x[1] - x[0]
    n = np.maximum(profile_impl_only, 1.0)
    g = Q * np.sum(hole_mobility(n) * n, axis=1) * dx
    return 1.0 / g
