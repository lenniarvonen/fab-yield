"""Variation + metrology layers: from a recipe to a wafer of dies to noisy data.

The hidden "truth" of a fab (furnace offsets, implanter calibration, radial
gradients) lives in `FabState`.  It is drawn once per seed and never exposed
to the agent.  `simulate_dies` turns a recipe into true per-die parameters
and electrical characteristics; `metrology` adds measurement noise.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np

from . import physics as P

# --------------------------------------------------------------------------
# Recipe knobs: name -> (min, max, default, unit, description)
# --------------------------------------------------------------------------
KNOBS = {
    "implant_dose": (1.0e11, 1.0e13, 1.0e12, "cm^-2", "Boron Vt-adjust implant dose"),
    "implant_energy": (5.0, 100.0, 20.0, "keV", "Boron implant energy (sets depth Rp)"),
    "anneal_temp": (800.0, 1100.0, 950.0, "degC", "Activation anneal furnace setpoint"),
    "anneal_time": (1.0, 120.0, 30.0, "min", "Activation anneal time"),
    "ox_temp": (800.0, 1050.0, 900.0, "degC", "Dry gate-oxidation furnace setpoint"),
    "ox_time": (5.0, 180.0, 45.0, "min", "Dry gate-oxidation time"),
    "gate_length": (0.25, 2.0, 1.0, "um", "Drawn gate length L"),
}
DEFAULT_RECIPE = {k: v[2] for k, v in KNOBS.items()}

# --------------------------------------------------------------------------
# Wafer geometry: 100 mm wafer, 9 mm die pitch, 3 mm edge exclusion
# --------------------------------------------------------------------------
WAFER_RADIUS_MM = 50.0
EDGE_EXCLUSION_MM = 3.0
DIE_PITCH_MM = 9.0


def _die_grid():
    r_ok = WAFER_RADIUS_MM - EDGE_EXCLUSION_MM
    centers = []
    n = int(np.ceil(WAFER_RADIUS_MM / DIE_PITCH_MM)) + 1
    for iy in range(-n, n):
        for ix in range(-n, n):
            x0, y0 = ix * DIE_PITCH_MM, iy * DIE_PITCH_MM
            corners = [(x0, y0), (x0 + DIE_PITCH_MM, y0), (x0, y0 + DIE_PITCH_MM), (x0 + DIE_PITCH_MM, y0 + DIE_PITCH_MM)]
            if all(np.hypot(cx, cy) <= r_ok for cx, cy in corners):
                centers.append((x0 + DIE_PITCH_MM / 2, y0 + DIE_PITCH_MM / 2))
    # Sort for a stable, human-friendly die numbering (row by row, top to bottom)
    centers.sort(key=lambda c: (-c[1], c[0]))
    return np.array(centers)


DIE_XY_MM = _die_grid()
N_DIES = len(DIE_XY_MM)
DIE_R_NORM = np.hypot(DIE_XY_MM[:, 0], DIE_XY_MM[:, 1]) / WAFER_RADIUS_MM  # 0 at centre, ~0.94 at edge

# --------------------------------------------------------------------------
# Documented process variation (known to the agent via docs/variation.md)
# --------------------------------------------------------------------------
SIGMA_FURNACE_RUN_C = 1.0  # run-to-run furnace temperature noise (each furnace)
SIGMA_DOSE_RUN = 0.01  # run-to-run implanter dose noise (relative)
SIGMA_DOSE_DIE = 0.015  # within-wafer dose non-uniformity (relative)
SIGMA_TOX_DIE = 0.01  # within-wafer oxide non-uniformity (relative)
SIGMA_L_DIE_UM = 0.02  # lithography/etch gate-length variation (absolute)
SIGMA_QF_DIE = 1.0e10  # fixed oxide charge variation, q/cm^2
TOX_RADIAL_NOMINAL = 0.02  # oxide is 2% thicker at the edge (r=1) than the centre
L_RADIAL_NOMINAL_UM = 0.02  # gates print 0.02 um longer at the edge

# Metrology noise
SIGMA_ELLIPS_NM = 0.15
SIGMA_4PP_REL = 0.01
SIGMA_VT_MEAS = 0.003  # V
SIGMA_ION_MEAS_REL = 0.005
SIGMA_IOFF_MEAS_LOG10 = 0.03


@dataclass
class FabState:
    """Hidden per-seed state of the fab (never shown to the agent)."""

    anneal_offset_c: float = 0.0  # true temp = setpoint + offset
    ox_offset_c: float = 0.0
    dose_factor: float = 1.0  # delivered dose = setpoint * factor
    tox_radial: float = TOX_RADIAL_NOMINAL
    l_radial_um: float = L_RADIAL_NOMINAL_UM
    noise_scale: float = 1.0  # multiplies all random variation

    def to_dict(self):
        return asdict(self)


def clamp_recipe(recipe):
    """Fill missing knobs with defaults and clamp to physical ranges."""
    out = {}
    for k, (lo, hi, default, _, _) in KNOBS.items():
        v = float(recipe.get(k, default))
        out[k] = float(np.clip(v, lo, hi))
    return out


def simulate_dies(recipe, state: FabState, rng: np.random.Generator, n_wafers=1):
    """Simulate `n_wafers` wafers processed with `recipe` in fab `state`.

    Returns a dict of arrays with shape (n_wafers, N_DIES): true tox_nm,
    vt, ion, ioff, rs (monitor-wafer sheet resistance), plus the true
    per-die gate length.
    """
    r = clamp_recipe(recipe)
    s = state.noise_scale
    nw, nd = n_wafers, N_DIES

    # Wafer-level (run-to-run) variation
    t_ann = r["anneal_temp"] + state.anneal_offset_c + s * SIGMA_FURNACE_RUN_C * rng.standard_normal(nw)
    t_ox = r["ox_temp"] + state.ox_offset_c + s * SIGMA_FURNACE_RUN_C * rng.standard_normal(nw)
    dose_w = r["implant_dose"] * state.dose_factor * (1.0 + s * SIGMA_DOSE_RUN * rng.standard_normal(nw))

    # Die-level variation
    rr = np.broadcast_to(DIE_R_NORM, (nw, nd))
    tox0 = P.oxide_thickness(t_ox, r["ox_time"])[:, None]
    tox = tox0 * (1.0 + state.tox_radial * rr**2) * (1.0 + s * SIGMA_TOX_DIE * rng.standard_normal((nw, nd)))
    dose = dose_w[:, None] * (1.0 + s * SIGMA_DOSE_DIE * rng.standard_normal((nw, nd)))
    lg_um = r["gate_length"] + state.l_radial_um * rr**2 + s * SIGMA_L_DIE_UM * rng.standard_normal((nw, nd))
    lg_um = np.maximum(lg_um, 0.05)
    qf = P.QF_NOMINAL + s * SIGMA_QF_DIE * rng.standard_normal((nw, nd))

    dt = P.thermal_budget(t_ann, r["anneal_time"], t_ox, r["ox_time"])[:, None] * np.ones((1, nd))
    consumed = P.SI_CONSUMED_PER_OX * np.maximum(tox - P.NATIVE_OXIDE_CM, 0.0)

    flat = lambda a: np.asarray(a).reshape(-1)
    energy = np.full(nw * nd, r["implant_energy"])
    prof = P.channel_profile(flat(dose), energy, flat(dt), flat(consumed))
    dev = P.device_characteristics(flat(tox), prof, flat(lg_um) * 1e-4, qf=flat(qf))

    # Monitor wafer for four-point probe: no oxide consumption beyond the
    # same thermal history; n-type substrate isolates the boron layer.
    rs = P.sheet_resistance(prof - P.N_SUB)

    shape = (nw, nd)
    return {
        "tox_nm": tox * 1e7,
        "vt": dev["vt"].reshape(shape),
        "ion": dev["ion"].reshape(shape),
        "ioff": dev["ioff"].reshape(shape),
        "rs": rs.reshape(shape),
        "gate_length_um": lg_um,
    }


def measure(true_wafer, tool, die_ids, rng):
    """Noisy metrology on one wafer (dict of 1-D arrays over dies)."""
    die_ids = np.asarray(die_ids, dtype=int)
    rows = []
    if tool == "ellipsometer":
        vals = true_wafer["tox_nm"][die_ids] + SIGMA_ELLIPS_NM * rng.standard_normal(len(die_ids))
        for d, v in zip(die_ids, vals):
            rows.append({"die": int(d), "tox_nm": round(float(v), 3)})
    elif tool == "four_point_probe":
        vals = true_wafer["rs"][die_ids] * (1.0 + SIGMA_4PP_REL * rng.standard_normal(len(die_ids)))
        for d, v in zip(die_ids, vals):
            rows.append({"die": int(d), "sheet_res_ohm_sq": round(float(v), 1)})
    elif tool == "electrical":
        n = len(die_ids)
        vt = true_wafer["vt"][die_ids] + SIGMA_VT_MEAS * rng.standard_normal(n)
        ion = true_wafer["ion"][die_ids] * (1.0 + SIGMA_ION_MEAS_REL * rng.standard_normal(n))
        ioff = true_wafer["ioff"][die_ids] * 10.0 ** (SIGMA_IOFF_MEAS_LOG10 * rng.standard_normal(n))
        for d, a, b, c in zip(die_ids, vt, ion, ioff):
            rows.append(
                {
                    "die": int(d),
                    "vt_V": round(float(a), 4),
                    "ion_mA": round(float(b) * 1e3, 4),
                    "ioff_pA": float(f"{float(c) * 1e12:.4g}"),
                }
            )
    else:
        raise ValueError(f"unknown tool {tool!r}")
    for row in rows:
        x, y = DIE_XY_MM[row["die"]]
        row["x_mm"], row["y_mm"] = round(float(x), 1), round(float(y), 1)
    return rows


def predict_batch(recipes, state: FabState, r_norm=0.6):
    """Noise-free prediction for many recipes at once (one die per recipe).

    `recipes` is a dict knob -> 1-D array (all the same length).  Used by the
    reference solver for fast coarse searches; it is the documented model with
    a given (estimated) fab state, nothing hidden.
    """
    n = len(next(iter(recipes.values())))
    r = {k: np.clip(np.asarray(recipes.get(k, np.full(n, v[2])), float), v[0], v[1]) for k, v in KNOBS.items()}
    t_ann = r["anneal_temp"] + state.anneal_offset_c
    t_ox = r["ox_temp"] + state.ox_offset_c
    tox = P.oxide_thickness(t_ox, r["ox_time"]) * (1.0 + state.tox_radial * r_norm**2)
    dose = r["implant_dose"] * state.dose_factor
    lg_um = r["gate_length"] + state.l_radial_um * r_norm**2
    dt = P.thermal_budget(t_ann, r["anneal_time"], t_ox, r["ox_time"])
    consumed = P.SI_CONSUMED_PER_OX * np.maximum(tox - P.NATIVE_OXIDE_CM, 0.0)
    prof = P.channel_profile(dose, r["implant_energy"], dt, consumed)
    dev = P.device_characteristics(tox, prof, lg_um * 1e-4)
    return {"tox_nm": tox * 1e7, "vt": dev["vt"], "ion": dev["ion"], "ioff": dev["ioff"],
            "rs": P.sheet_resistance(prof - P.N_SUB)}
