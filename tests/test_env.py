import itertools
import json
import math

import numpy as np
import pytest

from fab_yield import physics as P
from fab_yield import simulator as S
from fab_yield.baselines import do_nothing, stop_early
from fab_yield.env import LEVELS, MAX_ACTIONS, FabEnv


# ------------------------------------------------------------ physics sanity
def test_deal_grove_reference_point():
    assert P.oxide_thickness(900.0, 60.0) * 1e7 == pytest.approx(10.7, abs=0.1)
    # thicker with time and temperature
    assert P.oxide_thickness(950, 60) > P.oxide_thickness(900, 60) > P.oxide_thickness(900, 30)


def test_boron_diffusivity_textbook_values():
    assert P.boron_diffusivity(1000.0) == pytest.approx(1.5e-14, rel=0.1)
    assert P.boron_diffusivity(1100.0) == pytest.approx(1.5e-13, rel=0.15)


def test_profile_conserves_dose():
    prof = P.channel_profile(np.array([1e12]), np.array([20.0]), np.array([1e-12]), np.array([0.0]))
    dx = P.X_GRID_CM[1] - P.X_GRID_CM[0]
    assert np.trapezoid(prof[0] - P.N_SUB, dx=dx) == pytest.approx(1e12, rel=0.01)


def _nominal(**kw):
    rec = dict(S.DEFAULT_RECIPE, **kw)
    return S.predict_batch({k: np.array([v]) for k, v in rec.items()}, S.FabState())


def test_device_trends():
    lo, hi = _nominal(implant_dose=1e12), _nominal(implant_dose=3e12)
    assert hi["vt"][0] > lo["vt"][0] and hi["ioff"][0] < lo["ioff"][0] and hi["ion"][0] < lo["ion"][0]
    short, long_ = _nominal(gate_length=0.3), _nominal(gate_length=1.5)
    assert short["vt"][0] < long_["vt"][0]  # roll-off
    assert short["ion"][0] > long_["ion"][0]
    hot = _nominal(anneal_temp=1080, implant_dose=3e12)
    assert hot["vt"][0] < hi["vt"][0]  # more diffusion -> less dose in depletion region


# -------------------------------------------------------------- determinism
def _script(env):
    out = [env.step({"tool": "run_wafer", "args": {"recipe": {"implant_dose": 3e12}}})]
    out.append(env.step({"tool": "measure", "args": {"wafer_id": 0, "tool": "electrical", "dies": 7}}))
    out.append(env.step({"tool": "measure", "args": {"wafer_id": 0, "tool": "ellipsometer", "dies": [0, 30, 67]}}))
    out.append(env.step({"tool": "submit_recipe", "args": {"recipe": {"implant_dose": 4.3e12}}}))
    return json.dumps(out, default=float)


@pytest.mark.parametrize("level", [1, 2, 3])
def test_deterministic(level):
    assert _script(FabEnv(level, 7)) == _script(FabEnv(level, 7))


def test_seeds_differ_in_level3():
    assert FabEnv(3, 0)._hidden_state() != FabEnv(3, 1)._hidden_state()


# --------------------------------------------------------- reward integrity
@pytest.mark.parametrize("level,seed", list(itertools.product([1, 2, 3], range(5))))
def test_baselines_score_zero(level, seed):
    assert do_nothing(FabEnv(level, seed))["score"] == 0.0
    assert stop_early(FabEnv(level, seed))["score"] < 0.01


def test_corner_recipes_do_not_yield():
    """Extreme recipes (all 128 corners of the knob box) never pass."""
    env = FabEnv(2, 0)
    best = 0.0
    for corner in itertools.product(*[(v[0], v[1]) for v in S.KNOBS.values()]):
        rec = dict(zip(S.KNOBS, corner))
        d = S.simulate_dies(rec, env._state, np.random.default_rng(0), 1)
        from fab_yield.env import in_spec
        best = max(best, in_spec(d).mean())
    assert best == 0.0


def test_hidden_state_not_leaked():
    env = FabEnv(3, 0)
    hidden = env._hidden_state()
    text = env.task_prompt() + json.dumps(env.info())
    for k in ("anneal_offset_c", "ox_offset_c", "dose_factor"):
        assert f"{hidden[k]:.3f}" not in text and k not in text


def test_submit_ends_episode_once():
    env = FabEnv(1, 0)
    r1 = env.step({"tool": "submit_recipe", "args": {"recipe": {"implant_dose": 4.3e12}}})
    r2 = env.step({"tool": "submit_recipe", "args": {"recipe": {"implant_dose": 4.3e12}}})
    assert r1["ok"] and not r2["ok"] and env.done
    assert env.result()["lot_yield"] == r1["result"]["lot_yield"]


@pytest.mark.parametrize(
    "recipe",
    [
        {"implant_dose": 1e20},
        {"implant_dose": float("nan")},
        {"implant_dose": "4e12"},
        {"implant_dose": True},
        {"bogus": 1.0},
        {"gate_length": 0.0},
        "not a dict",
    ],
)
def test_invalid_recipes_rejected_without_cost(recipe):
    env = FabEnv(2, 0)
    r = env.step({"tool": "run_wafer", "args": {"recipe": recipe}})
    assert not r["ok"] and env.spent == 0 and not env.wafers


def test_fixed_knobs_enforced_in_level1():
    env = FabEnv(1, 0)
    r = env.step({"tool": "run_wafer", "args": {"recipe": {"gate_length": 1.5}}})
    assert not r["ok"] and "fixed" in r["error"]
    ok = env.step({"tool": "run_wafer", "args": {"recipe": {"gate_length": 0.47, "implant_dose": 4e12}}})
    assert ok["ok"]


def test_budget_enforced():
    env = FabEnv(1, 0)
    for _ in range(int(LEVELS[1].budget)):
        assert env.step({"tool": "run_wafer", "args": {"recipe": {}}})["ok"]
    r = env.step({"tool": "run_wafer", "args": {"recipe": {}}})
    assert not r["ok"] and "budget" in r["error"]
    r = env.step({"tool": "measure", "args": {"wafer_id": 0, "tool": "electrical", "dies": 1}})
    assert not r["ok"]


def test_measure_validation():
    env = FabEnv(2, 0)
    assert not env.step({"tool": "measure", "args": {"wafer_id": 0, "tool": "electrical"}})["ok"]
    env.step({"tool": "run_wafer", "args": {"recipe": {}}})
    for bad in [{"wafer_id": 0, "tool": "xray"}, {"wafer_id": 0, "tool": "electrical", "dies": [999]},
                {"wafer_id": 0, "tool": "electrical", "dies": 0}, {"wafer_id": True, "tool": "electrical"}]:
        assert not env.step({"tool": "measure", "args": bad})["ok"]
    assert env.spent == 1.0


def test_action_limit_ends_episode_with_zero():
    env = FabEnv(1, 0)
    for _ in range(MAX_ACTIONS):
        env.step({"tool": "read_docs", "args": {"topic": "index"}})
    assert env.done and env.result()["score"] == 0.0


def test_all_docs_readable():
    env = FabEnv(1, 0)
    idx = env.read_docs("index")["text"]
    for topic in ["process_flow", "oxidation", "implant", "diffusion", "device", "variation", "metrology", "scoring"]:
        assert f"`{topic}`" in idx
        assert len(env.read_docs(topic)["text"]) > 200


def test_measurement_noise_is_unbiased_and_documented_size():
    env = FabEnv(2, 0)
    env.run_wafer({})
    truth = env.wafers[0].truth["tox_nm"][10]
    vals = [env.measure(0, "ellipsometer", [10])["data"][0]["tox_nm"] for _ in range(200)]
    assert np.mean(vals) == pytest.approx(truth, abs=0.05)
    assert np.std(vals) == pytest.approx(S.SIGMA_ELLIPS_NM, rel=0.2)
