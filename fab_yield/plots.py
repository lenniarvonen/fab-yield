"""Human-facing figures (agents never see these).

    python -m fab_yield.plots --level 3 --seed 0      -> results/figures/*.png

Runs the reference policy on (level, seed) and draws:
  * wafer map of the submitted recipe (pass/fail and Vt)
  * channel doping profile, default vs submitted recipe
  * process window: true yield over dose x anneal temperature, with the
    experiment path (every test wafer) and the submitted recipe overlaid
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from matplotlib.patches import Patch, Rectangle  # noqa: E402

from . import physics as P  # noqa: E402
from . import simulator as S  # noqa: E402
from .env import SPEC, FabEnv, in_spec  # noqa: E402

INK, INK2, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#fcfcfb"
GOOD, CRITICAL = "#0ca30c", "#d03b3b"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]
BLUES = LinearSegmentedColormap.from_list("blues", ["#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"])

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": "#c3c2b7", "axes.labelcolor": INK2, "xtick.color": MUTED, "ytick.color": MUTED,
    "text.color": INK, "axes.grid": False, "font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
})


def wafer_map(env: FabEnv, path: Path):
    lot = {k: v[0] for k, v in env._lot.items()}
    ok = in_spec({k: v for k, v in lot.items()})
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.8))
    h = S.DIE_PITCH_MM / 2
    for ax in axes:
        ax.add_patch(plt.Circle((0, 0), S.WAFER_RADIUS_MM, fill=False, color="#c3c2b7", lw=1.5))
        ax.set_xlim(-52, 52); ax.set_ylim(-52, 52); ax.set_aspect("equal"); ax.axis("off")
    for (x, y), good in zip(S.DIE_XY_MM, ok):
        axes[0].add_patch(Rectangle((x - h + 0.4, y - h + 0.4), 2 * h - 0.8, 2 * h - 0.8,
                                    facecolor=GOOD if good else CRITICAL, edgecolor=SURFACE,
                                    hatch=None if good else "///", lw=0))
    axes[0].legend(handles=[Patch(facecolor=GOOD, label=f"pass ({ok.sum()})"),
                            Patch(facecolor=CRITICAL, hatch="///", label=f"fail ({(~ok).sum()})")],
                   loc="lower center", bbox_to_anchor=(0.5, -0.08), ncol=2, frameon=False)
    axes[0].set_title(f"Production wafer 1: yield {ok.mean():.0%}", color=INK)
    vmin, vmax = SPEC["vt_min_V"] - 0.05, SPEC["vt_max_V"] + 0.05
    for (x, y), v in zip(S.DIE_XY_MM, lot["vt"]):
        c = BLUES((np.clip(v, vmin, vmax) - vmin) / (vmax - vmin))
        axes[1].add_patch(Rectangle((x - h + 0.4, y - h + 0.4), 2 * h - 0.8, 2 * h - 0.8, facecolor=c, lw=0))
    sm = plt.cm.ScalarMappable(cmap=BLUES, norm=plt.Normalize(vmin, vmax))
    cb = fig.colorbar(sm, ax=axes[1], fraction=0.04, pad=0.02)
    cb.set_label("Vt (V)"); cb.ax.axhline(SPEC["vt_min_V"], color=INK, lw=1); cb.ax.axhline(SPEC["vt_max_V"], color=INK, lw=1)
    axes[1].set_title(f"Vt map (spec {SPEC['vt_min_V']}–{SPEC['vt_max_V']} V)", color=INK)
    fig.suptitle(f"Level {env.level.level}, seed {env.seed}: submitted recipe", color=INK2, fontsize=10)
    fig.savefig(path, dpi=150, bbox_inches="tight"); plt.close(fig)


def _profile(recipe, state=S.FabState()):
    r = S.clamp_recipe(recipe)
    t_ann, t_ox = r["anneal_temp"] + state.anneal_offset_c, r["ox_temp"] + state.ox_offset_c
    tox = P.oxide_thickness(t_ox, r["ox_time"])
    dt = P.thermal_budget(t_ann, r["anneal_time"], t_ox, r["ox_time"])
    cons = P.SI_CONSUMED_PER_OX * (tox - P.NATIVE_OXIDE_CM)
    prof = P.channel_profile(np.array([r["implant_dose"] * state.dose_factor]), np.array([r["implant_energy"]]),
                             np.array([dt]), np.array([cons]))
    w, _, _ = P.depletion(prof)
    return prof[0], w[0]


def doping_profile(recipes: dict, path: Path):
    fig, ax = plt.subplots(figsize=(7, 4.2))
    x_nm = P.X_GRID_CM * 1e7
    for (name, rec), col in zip(recipes.items(), SERIES):
        prof, w = _profile(rec)
        ax.semilogy(x_nm, prof, color=col, lw=2, label=name)
        ax.axvline(w * 1e7, color=col, lw=1, ls="--")
        ax.annotate(f"W_d = {w * 1e7:.0f} nm", (w * 1e7, prof.max() * 0.6), color=INK2, fontsize=8,
                    xytext=(4, 0), textcoords="offset points")
    ax.axhline(P.N_SUB, color=MUTED, lw=1, ls=":")
    ax.text(990, P.N_SUB * 1.2, "substrate 1e16", color=MUTED, ha="right", fontsize=8)
    ax.set_xlim(0, 1000); ax.set_xlabel("depth below gate oxide (nm)"); ax.set_ylabel("boron (cm$^{-3}$)")
    ax.grid(axis="y", color=GRID, lw=0.6); ax.legend(frameon=False)
    ax.set_title("Channel doping profile (nominal fab, dashed = depletion edge)", color=INK, fontsize=10)
    fig.savefig(path, dpi=150, bbox_inches="tight"); plt.close(fig)


def process_window(env: FabEnv, path: Path, n=31):
    rec = env.final["recipe"]
    state = env._state
    wd = [w.recipe["implant_dose"] for w in env.wafers] + [rec["implant_dose"]]
    wt = [w.recipe["anneal_temp"] for w in env.wafers] + [rec["anneal_temp"]]
    doses = np.geomspace(max(min(wd) / 1.5, 1e11), min(max(wd) * 1.5, 1e13), n)
    temps = np.linspace(max(min(wt) - 40, 800), min(max(wt) + 40, 1100), n)
    y = np.zeros((n, n))
    for i, t in enumerate(temps):
        for j, d in enumerate(doses):
            lot = S.simulate_dies(dict(rec, implant_dose=d, anneal_temp=t), state, np.random.default_rng(5), 2)
            y[i, j] = in_spec(lot).mean()
    fig, ax = plt.subplots(figsize=(7, 5))
    im = ax.pcolormesh(doses, temps, y, cmap=BLUES, vmin=0, vmax=1, shading="nearest")
    cb = fig.colorbar(im, ax=ax); cb.set_label("true yield (other knobs at submitted values)")
    for w in env.wafers:
        ax.plot(w.recipe["implant_dose"], w.recipe["anneal_temp"], "o", ms=9, mfc=SERIES[1], mec=SURFACE, mew=2)
        ax.annotate(f"wafer {w.wafer_id}", (w.recipe["implant_dose"], w.recipe["anneal_temp"]),
                    xytext=(6, 6), textcoords="offset points", fontsize=8, color=INK)
    ax.plot(rec["implant_dose"], rec["anneal_temp"], "*", ms=16, mfc=SERIES[2], mec=SURFACE, mew=1.5,
            label="submitted recipe")
    ax.plot([], [], "o", ms=9, mfc=SERIES[1], mec=SURFACE, label="test wafers")
    ax.set_xscale("log"); ax.set_xlabel("implant dose setpoint (cm$^{-2}$)"); ax.set_ylabel("anneal setpoint (°C)")
    ax.set_xlim(doses[0], doses[-1]); ax.set_ylim(temps[0], temps[-1])
    ax.legend(loc="lower right", frameon=True, facecolor=SURFACE, edgecolor=GRID, fontsize=8)
    ax.set_title(f"Process window, level {env.level.level} seed {env.seed} (hidden fab state applied)",
                 color=INK, fontsize=10)
    fig.savefig(path, dpi=150, bbox_inches="tight"); plt.close(fig)


def main(argv=None):
    from .reference import run_reference

    ap = argparse.ArgumentParser()
    ap.add_argument("--level", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="results/figures")
    a = ap.parse_args(argv)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    env = FabEnv(a.level, a.seed)
    run_reference(env)
    tag = f"L{a.level}_s{a.seed}"
    wafer_map(env, out / f"wafer_map_{tag}.png")
    process_window(env, out / f"process_window_{tag}.png")
    doping_profile({"default recipe": S.DEFAULT_RECIPE, "reference recipe": env.final["recipe"]},
                   out / f"doping_profile_{tag}.png")
    print(f"figures written to {out}/ (yield {env.final['lot_yield']:.3f})")


if __name__ == "__main__":
    main()
