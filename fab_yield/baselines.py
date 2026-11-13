"""Baseline policies the judges run: do-nothing, stop-early, open-loop physics."""

from __future__ import annotations

from . import simulator as S
from .env import FabEnv


def do_nothing(env: FabEnv) -> dict:
    """Takes no action and never submits."""
    return env.result()


def stop_early(env: FabEnv) -> dict:
    """Submits the default recipe immediately, without any experiment."""
    env.step({"tool": "submit_recipe", "args": {"recipe": {}}})
    return env.result()


def open_loop(env: FabEnv) -> dict:
    """Optimises with the documented model assuming a calibrated fab, spends
    nothing on wafers and submits. Strong in levels 1-2; level 3's hidden
    calibration errors are what it cannot see."""
    from .reference import RecipeOptimizer

    lv = env.level
    base = dict(S.DEFAULT_RECIPE)
    base.update(lv.fixed)
    best, _ = RecipeOptimizer(lv.free_knobs, base, S.FabState(noise_scale=lv.noise_scale), env.seed).solve()
    env.step({"tool": "submit_recipe", "args": {"recipe": {k: best[k] for k in lv.free_knobs}}})
    return env.result()
