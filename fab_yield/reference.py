"""Reference solver: physics-model calibration + robust recipe optimisation.

It plays through the same `env.step` interface as any agent and only uses
information an agent also has: the documented model (docs/ describe every
equation and constant used in `physics.py` / `simulator.py`), the spec, and
its own measurements.  It never reads the hidden fab state.

Strategy
    1. (Level 3 only) Run two deliberately different calibration wafers and
       measure oxide thickness, sheet resistance and Vt/Ion/Ioff.  Fit the
       four unknown fab parameters (anneal offset, oxidation offset, dose
       factor, oxide radial gradient) by least squares (Nelder-Mead).
    2. Coarse search: score 20k random recipes with the calibrated model by
       their worst normalised spec margin.
    3. Refine the best candidates with Monte Carlo yield (common random
       numbers, documented variation) plus a small centring bonus.
    4. Run one verification wafer at the chosen recipe, refit (level 3), and
       re-optimise locally if the fit moved.  Submit.
"""

from __future__ import annotations

import numpy as np

from . import simulator as S
from .env import SPEC, FabEnv

# ------------------------------------------------------------------ helpers
_MARGIN_SCALE = {"vt": 0.02, "ion": 0.1, "ioff": 0.15, "tox": 0.3}


def _margins(d):
    """Worst normalised spec margin per die (>0 means pass)."""
    m_vt = np.minimum(d["vt"] - SPEC["vt_min_V"], SPEC["vt_max_V"] - d["vt"]) / _MARGIN_SCALE["vt"]
    m_ion = (d["ion"] * 1e3 - SPEC["ion_min_mA"]) / _MARGIN_SCALE["ion"]
    m_ioff = np.log10(SPEC["ioff_max_pA"] / np.maximum(d["ioff"] * 1e12, 1e-30)) / _MARGIN_SCALE["ioff"]
    m_tox = (d["tox_nm"] - SPEC["tox_min_nm"]) / _MARGIN_SCALE["tox"]
    return np.minimum(np.minimum(m_vt, m_ion), np.minimum(m_ioff, m_tox))


def _to_unit(rec, free):
    out = []
    for k in free:
        lo, hi = S.KNOBS[k][0], S.KNOBS[k][1]
        v = rec[k]
        out.append((np.log(v) - np.log(lo)) / (np.log(hi) - np.log(lo)) if k == "implant_dose" else (v - lo) / (hi - lo))
    return np.array(out)


def _from_unit(z, free, base):
    rec = dict(base)
    for k, u in zip(free, np.clip(z, 0.0, 1.0)):
        lo, hi = S.KNOBS[k][0], S.KNOBS[k][1]
        v = np.exp(np.log(lo) + u * (np.log(hi) - np.log(lo))) if k == "implant_dose" else lo + u * (hi - lo)
        rec[k] = float(np.clip(v, lo, hi))
    return rec


def nelder_mead(f, x0, step, iters=400, tol=1e-8):
    """Minimal Nelder-Mead (numpy only)."""
    n = len(x0)
    pts = [np.array(x0, float)]
    for i in range(n):
        p = np.array(x0, float)
        p[i] += step[i]
        pts.append(p)
    vals = [f(p) for p in pts]
    for _ in range(iters):
        order = np.argsort(vals)
        pts = [pts[i] for i in order]
        vals = [vals[i] for i in order]
        if abs(vals[-1] - vals[0]) < tol:
            break
        c = np.mean(pts[:-1], axis=0)
        xr = c + (c - pts[-1])
        fr = f(xr)
        if fr < vals[0]:
            xe = c + 2.0 * (c - pts[-1])
            fe = f(xe)
            pts[-1], vals[-1] = (xe, fe) if fe < fr else (xr, fr)
        elif fr < vals[-2]:
            pts[-1], vals[-1] = xr, fr
        else:
            xc = c + 0.5 * (pts[-1] - c)
            fc = f(xc)
            if fc < vals[-1]:
                pts[-1], vals[-1] = xc, fc
            else:
                for i in range(1, len(pts)):
                    pts[i] = pts[0] + 0.5 * (pts[i] - pts[0])
                    vals[i] = f(pts[i])
    i = int(np.argmin(vals))
    return pts[i], vals[i]


# ------------------------------------------------------------- optimiser
class RecipeOptimizer:
    def __init__(self, free, base, state: S.FabState, seed=0, uncertainty=None):
        """`uncertainty` = 1-sigma of the calibration estimate
        (anneal_offset_c, ox_offset_c, relative dose): each Monte Carlo wafer
        draws its own plausible fab state, so the optimiser prefers recipes
        that are robust to calibration error, not just to process noise."""
        self.free = list(free)
        self.base = dict(base)
        self.state = state
        self.seed = seed
        self.n_wafers = 8
        rng = np.random.default_rng([seed, 98])
        self.states = []
        for _ in range(self.n_wafers):
            if uncertainty is None:
                self.states.append(state)
            else:
                ua, uo, ud = uncertainty
                self.states.append(S.FabState(
                    anneal_offset_c=state.anneal_offset_c + ua * rng.standard_normal(),
                    ox_offset_c=state.ox_offset_c + uo * rng.standard_normal(),
                    dose_factor=state.dose_factor * (1 + ud * rng.standard_normal()),
                    tox_radial=state.tox_radial, l_radial_um=state.l_radial_um, noise_scale=state.noise_scale))

    def objective(self, rec):
        rng = np.random.default_rng([self.seed, 99])  # common random numbers
        if all(st is self.state for st in self.states):
            d = S.simulate_dies(rec, self.state, rng, n_wafers=self.n_wafers)
        else:
            parts = [S.simulate_dies(rec, st, rng, n_wafers=1) for st in self.states]
            d = {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}
        m = _margins(d)
        return float((m > 0).mean() + 0.02 * np.clip(m, -2, 3).mean())

    def coarse(self, n=20000, top=4):
        rng = np.random.default_rng([self.seed, 7])
        z = rng.random((n, len(self.free)))
        recs = {k: np.empty(n) for k in self.free}
        for j, k in enumerate(self.free):
            lo, hi = S.KNOBS[k][0], S.KNOBS[k][1]
            recs[k] = np.exp(np.log(lo) + z[:, j] * np.log(hi / lo)) if k == "implant_dose" else lo + z[:, j] * (hi - lo)
        for k, v in self.base.items():
            if k not in self.free:
                recs[k] = np.full(n, v)
        scores = np.empty(n)
        for s in range(0, n, 2000):
            sub = {k: v[s : s + 2000] for k, v in recs.items()}
            scores[s : s + 2000] = _margins(S.predict_batch(sub, self.state))
        idx = np.argsort(-scores)[:top]
        return [_from_unit(z[i], self.free, self.base) for i in idx]

    def refine(self, start, iters=250, step=0.08):
        rng = np.random.default_rng([self.seed, 11])
        best, by = dict(start), self.objective(start)
        z = _to_unit(best, self.free)
        for it in range(iters):
            cand_z = np.clip(z + step * rng.standard_normal(len(z)), 0, 1)
            cand = _from_unit(cand_z, self.free, self.base)
            y = self.objective(cand)
            if y >= by:
                best, by, z = cand, y, cand_z
            if it % 80 == 79:
                step *= 0.5
        return best, by

    def solve(self, starts=None):
        cands = list(starts or []) + self.coarse()
        results = [self.refine(c, iters=150) for c in cands]
        best, by = max(results, key=lambda t: t[1])
        return self.refine(best, iters=150, step=0.03)


# ------------------------------------------------------------- calibration
class Calibrator:
    """Fit the unknown fab parameters to measurements.

    The oxide radial gradient comes from a linear regression of ellipsometer
    sites on r^2.  The three offsets come from a least-squares fit of
    per-wafer means (wafer means average out die noise but not run-to-run
    furnace/implanter noise, so their residuals are weighted by documented
    run-to-run sigmas) with a weak prior centred on "calibrated".
    """

    # 1-sigma of a wafer-mean residual (run-to-run noise dominated)
    SIG = {"tox_nm": 0.12, "rs": 0.02, "vt": 0.015, "ion": 0.01}
    PRIOR = np.array([15.0, 10.0, 0.08])

    def __init__(self):
        self.obs = []  # (recipe, die_ids, quantity, values)

    def add(self, recipe, tool, rows):
        ids = np.array([r["die"] for r in rows])
        if tool == "ellipsometer":
            self.obs.append((recipe, ids, "tox_nm", np.array([r["tox_nm"] for r in rows])))
        elif tool == "four_point_probe":
            self.obs.append((recipe, ids, "rs", np.array([r["sheet_res_ohm_sq"] for r in rows])))
        elif tool == "electrical":
            self.obs.append((recipe, ids, "vt", np.array([r["vt_V"] for r in rows])))
            self.obs.append((recipe, ids, "ion", np.array([r["ion_mA"] for r in rows]) * 1e-3))

    def radial(self):
        gs = []
        for _, ids, q, vals in self.obs:
            if q == "tox_nm" and len(ids) >= 3:
                r2 = S.DIE_R_NORM[ids] ** 2
                slope, icpt = np.polyfit(r2, vals, 1)
                gs.append(slope / icpt)
        return float(np.clip(np.mean(gs), 0.0, 0.1)) if gs else S.TOX_RADIAL_NOMINAL

    def loss(self, theta, g):
        a, o, lf = theta
        st = S.FabState(anneal_offset_c=a, ox_offset_c=o, dose_factor=float(np.exp(lf)), tox_radial=g)
        total = float(np.sum((np.asarray(theta) / self.PRIOR) ** 2))
        for recipe, ids, q, vals in self.obs:
            n = len(ids)
            recs = {k: np.full(n, v) for k, v in recipe.items()}
            pred = S.predict_batch(recs, st, r_norm=S.DIE_R_NORM[ids])[q]
            if q in ("rs", "ion"):
                res = (np.mean(np.log(vals)) - np.mean(np.log(pred))) / self.SIG[q]
            else:
                res = (np.mean(vals) - np.mean(pred)) / self.SIG[q]
            total += res**2
        return total

    def fit(self, x0=(0.0, 0.0, 0.0)):
        g = self.radial()
        best = None
        for start in [x0, (12.0, 6.0, 0.06), (-12.0, -6.0, -0.06), (12.0, -6.0, -0.06), (-12.0, 6.0, 0.06)]:
            x, v = nelder_mead(lambda t: self.loss(t, g), np.array(start[:3], float), step=np.array([6.0, 4.0, 0.04]), iters=300)
            if best is None or v < best[1]:
                best = (x, v)
        a, o, lf = best[0]
        st = S.FabState(anneal_offset_c=float(a), ox_offset_c=float(o), dose_factor=float(np.exp(lf)), tox_radial=g)
        return st, np.array([a, o, lf, g])


# ------------------------------------------------------------------ runner
def _act(env: FabEnv, _name, **args):
    res = env.step({"tool": _name, "args": args})
    if not res["ok"]:
        raise RuntimeError(f"reference action failed: {res['error']}")
    return res["result"]


def _characterise(env, cal, recipe, n_elec=12):
    w = _act(env, "run_wafer", recipe=recipe)
    wid, full = w["wafer_id"], w["recipe"]
    for tool, n in (("ellipsometer", 9), ("four_point_probe", 5), ("electrical", n_elec)):
        m = _act(env, "measure", wafer_id=wid, tool=tool, dies=n)
        cal.add(full, tool, m["data"])
    return full


def run_reference(env: FabEnv, verbose=False) -> dict:
    lv = env.level
    free = [k for k in lv.free_knobs]
    base = dict(S.DEFAULT_RECIPE)
    base.update(lv.fixed)
    sub = lambda rec: {k: rec[k] for k in free}

    state = S.FabState(noise_scale=lv.noise_scale)
    cal = Calibrator()
    if lv.hidden_offsets:
        # Two calibration wafers with different thermal budgets and doses so
        # that dose factor and anneal offset are separable.
        nominal_best, _ = RecipeOptimizer(free, base, state, env.seed).solve()
        # Short-loop monitors: a near-zero-diffusion anneal isolates the dose
        # factor; a hot, long anneal makes Vt sensitive to the anneal offset.
        cold = dict(nominal_best, anneal_temp=850.0, anneal_time=10.0)
        hot = dict(nominal_best, anneal_temp=1050.0, anneal_time=60.0)
        _characterise(env, cal, sub(cold))
        _characterise(env, cal, sub(hot))
        est, theta = cal.fit()
        est.noise_scale = lv.noise_scale
        state = est
        if verbose:
            print("calibrated:", np.round(theta, 4), "truth:", env._hidden_state())
        unc = (6.0, 2.0, 0.02)
        best, _ = RecipeOptimizer(free, base, state, env.seed, uncertainty=unc).solve(starts=[nominal_best])
        # Verify at the chosen recipe, refit with all data, polish.
        _characterise(env, cal, sub(best))
        est, theta = cal.fit(x0=tuple(theta[:3]))
        est.noise_scale = lv.noise_scale
        opt = RecipeOptimizer(free, base, est, env.seed, uncertainty=(4.0, 1.5, 0.015))
        best, _ = opt.refine(best, iters=150, step=0.03)
        if verbose:
            print("refit:", np.round(theta, 4))
    else:
        # Calibrated fab: the documented model is exact up to random
        # variation, so the best play is to optimise and submit without
        # spending budget on test wafers.
        best, _ = RecipeOptimizer(free, base, state, env.seed).solve()
    _act(env, "submit_recipe", recipe=sub(best))
    return env.result()
