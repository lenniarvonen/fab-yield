"""Fab Yield environment: the agent-facing API, levels and scoring.

Four tools (see `tools.py` for JSON schemas):
    run_wafer(recipe)                 -> wafer_id      (costs 1.0 budget unit)
    measure(wafer_id, tool, dies)     -> noisy data    (costs per site)
    read_docs(topic)                  -> documentation (free)
    submit_recipe(recipe)             -> ends episode; production lot is scored

Score = max(0, lot_yield - COST_WEIGHT * budget_spent).  No submission -> 0.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from . import simulator as S

DOCS_DIR = Path(__file__).parent / "docs"

SPEC = {
    "vt_min_V": 0.50,
    "vt_max_V": 0.60,
    "ion_min_mA": 3.5,
    "ioff_max_pA": 20.0,
    "tox_min_nm": 7.0,
}
COSTS = {
    "run_wafer": 1.0,
    "electrical": 0.02,  # per die
    "ellipsometer": 0.005,  # per site
    "four_point_probe": 0.005,  # per site
}
COST_WEIGHT = 0.005  # score penalty per budget unit spent
LOT_WAFERS = 15  # production lot used for the final yield (15 x 68 = 1020 dies)
MAX_ACTIONS = 100


@dataclass(frozen=True)
class Level:
    level: int
    name: str
    free_knobs: tuple
    fixed: dict
    budget: float
    noise_scale: float
    hidden_offsets: bool
    blurb: str


_L1_FIXED = {
    "implant_energy": 20.0,
    "anneal_temp": 980.0,
    "anneal_time": 90.0,
    "ox_temp": 900.0,
    "ox_time": 60.0,
    "gate_length": 0.47,
}

LEVELS = {
    1: Level(
        1,
        "single knob",
        ("implant_dose",),
        _L1_FIXED,
        budget=6.0,
        noise_scale=0.5,
        hidden_offsets=False,
        blurb="Only the implant dose is free; every other step of the recipe is frozen. Tools are calibrated.",
    ),
    2: Level(
        2,
        "coupled knobs",
        tuple(S.KNOBS),
        {},
        budget=20.0,
        noise_scale=1.0,
        hidden_offsets=False,
        blurb="All seven knobs are free and interact (thermal budget, oxide, gate length). Tools are calibrated.",
    ),
    3: Level(
        3,
        "miscalibrated fab",
        tuple(S.KNOBS),
        {},
        budget=20.0,
        noise_scale=1.0,
        hidden_offsets=True,
        blurb=(
            "All seven knobs are free. The furnaces and the implanter in this fab are NOT calibrated: "
            "real temperatures and delivered dose differ from the setpoints by unknown, constant amounts, "
            "and the oxide radial non-uniformity is unknown. Detect and compensate using measurements."
        ),
    ),
}


def sample_fab_state(level: Level, seed: int) -> S.FabState:
    """Hidden per-seed fab state. Deterministic in (level, seed)."""
    if not level.hidden_offsets:
        return S.FabState(noise_scale=level.noise_scale)
    rng = np.random.default_rng([seed, 0, level.level])
    sign = lambda: 1.0 if rng.random() < 0.5 else -1.0
    return S.FabState(
        anneal_offset_c=sign() * rng.uniform(8.0, 20.0),
        ox_offset_c=sign() * rng.uniform(5.0, 12.0),
        dose_factor=1.0 + sign() * rng.uniform(0.04, 0.10),
        tox_radial=rng.uniform(0.0, 0.04),
        noise_scale=level.noise_scale,
    )


def in_spec(dies) -> np.ndarray:
    """Boolean pass/fail per die against SPEC (true, not measured, values)."""
    return (
        (dies["vt"] >= SPEC["vt_min_V"])
        & (dies["vt"] <= SPEC["vt_max_V"])
        & (dies["ion"] * 1e3 >= SPEC["ion_min_mA"])
        & (dies["ioff"] * 1e12 <= SPEC["ioff_max_pA"])
        & (dies["tox_nm"] >= SPEC["tox_min_nm"])
    )


class EnvError(Exception):
    pass


@dataclass
class WaferRecord:
    wafer_id: int
    recipe: dict
    truth: dict  # 1-D arrays over dies (hidden from the agent)
    measurements: list = field(default_factory=list)


class FabEnv:
    """One episode of Fab Yield. Create a new instance per episode."""

    def __init__(self, level: int = 1, seed: int = 0):
        if level not in LEVELS:
            raise ValueError(f"level must be one of {sorted(LEVELS)}")
        self.level = LEVELS[level]
        self.seed = int(seed)
        self._state = sample_fab_state(self.level, self.seed)
        self.spent = 0.0
        self.actions = 0
        self.done = False
        self.wafers: list[WaferRecord] = []
        self._n_measure = 0
        self.final: dict | None = None
        self.log: list[dict] = []

    # ------------------------------------------------------------------ info
    def task_prompt(self) -> str:
        """Natural-language task statement for an LLM agent."""
        lv = self.level
        knobs = []
        for k in lv.free_knobs:
            lo, hi, d, unit, desc = S.KNOBS[k]
            knobs.append(f"  - {k} [{unit}]: {desc}. Range {lo:g} .. {hi:g}, default {d:g}")
        fixed = "".join(f"\n  - {k} = {v:g} (fixed)" for k, v in lv.fixed.items()) if lv.fixed else ""
        return (
            f"You are the process engineer for an NMOS transistor line (Fab Yield, level {lv.level}: {lv.name}).\n"
            f"{lv.blurb}\n\n"
            "Goal: submit ONE recipe for a production lot. Your score is the fraction of production dies that meet "
            f"spec, minus {COST_WEIGHT} per budget unit you spent on test wafers and measurements.\n"
            f"Spec per die: {SPEC['vt_min_V']} V <= Vt <= {SPEC['vt_max_V']} V, Ion >= {SPEC['ion_min_mA']} mA, "
            f"Ioff <= {SPEC['ioff_max_pA']} pA, gate oxide >= {SPEC['tox_min_nm']} nm.\n\n"
            f"Recipe knobs you control:\n" + "\n".join(knobs) + fixed + "\n\n"
            f"Budget: {lv.budget:g} units. Costs: run_wafer {COSTS['run_wafer']}, electrical test "
            f"{COSTS['electrical']}/die, ellipsometer {COSTS['ellipsometer']}/site, four-point probe "
            f"{COSTS['four_point_probe']}/site. read_docs is free. Each wafer has {S.N_DIES} dies (ids 0..{S.N_DIES - 1}).\n"
            f"You have at most {MAX_ACTIONS} tool calls. If you never call submit_recipe you score 0.\n"
            "Start with read_docs('index')."
        )

    def info(self) -> dict:
        return {
            "level": self.level.level,
            "seed": self.seed,
            "budget": self.level.budget,
            "budget_spent": round(self.spent, 4),
            "budget_remaining": round(self.level.budget - self.spent, 4),
            "actions_used": self.actions,
            "done": self.done,
        }

    # --------------------------------------------------------------- helpers
    def _check_recipe(self, recipe) -> dict:
        if not isinstance(recipe, dict):
            raise EnvError("recipe must be an object mapping knob names to numbers")
        out = {}
        for k, v in recipe.items():
            if k not in S.KNOBS:
                raise EnvError(f"unknown knob {k!r}; knobs are {list(S.KNOBS)}")
            if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v):
                raise EnvError(f"knob {k!r} must be a finite number")
            lo, hi = S.KNOBS[k][0], S.KNOBS[k][1]
            if not lo <= v <= hi:
                raise EnvError(f"knob {k!r}={v} outside allowed range [{lo:g}, {hi:g}]")
            if k not in self.level.free_knobs and float(v) != self.level.fixed.get(k):
                raise EnvError(f"knob {k!r} is fixed at {self.level.fixed.get(k)} in this level")
            out[k] = float(v)
        full = dict(S.DEFAULT_RECIPE)
        full.update(self.level.fixed)
        full.update(out)
        return full

    def _charge(self, cost):
        if self.spent + cost > self.level.budget + 1e-9:
            raise EnvError(
                f"insufficient budget: need {cost:g}, remaining {self.level.budget - self.spent:g}"
            )
        self.spent += cost

    # ----------------------------------------------------------------- tools
    def run_wafer(self, recipe: dict) -> dict:
        full = self._check_recipe(recipe)
        self._charge(COSTS["run_wafer"])
        wid = len(self.wafers)
        rng = np.random.default_rng([self.seed, 1, self.level.level, wid])
        truth = S.simulate_dies(full, self._state, rng, n_wafers=1)
        truth = {k: v[0] for k, v in truth.items()}
        self.wafers.append(WaferRecord(wid, full, truth))
        return {"wafer_id": wid, "recipe": full, "n_dies": S.N_DIES}

    def measure(self, wafer_id, tool: str, dies=None) -> dict:
        if isinstance(wafer_id, bool) or not isinstance(wafer_id, int) or not 0 <= wafer_id < len(self.wafers):
            raise EnvError(f"no wafer with id {wafer_id!r}; run_wafer first")
        if tool not in ("electrical", "ellipsometer", "four_point_probe"):
            raise EnvError("tool must be 'electrical', 'ellipsometer' or 'four_point_probe'")
        if dies is None:
            dies = 5
        if isinstance(dies, bool):
            raise EnvError("dies must be an integer count or a list of die ids")
        if isinstance(dies, int):
            if not 1 <= dies <= S.N_DIES:
                raise EnvError(f"dies count must be in 1..{S.N_DIES}")
            ids = np.unique(np.round(np.linspace(0, S.N_DIES - 1, dies)).astype(int)).tolist()
        elif isinstance(dies, list) and dies:
            if not all(isinstance(d, int) and not isinstance(d, bool) and 0 <= d < S.N_DIES for d in dies):
                raise EnvError(f"die ids must be integers in 0..{S.N_DIES - 1}")
            if len(dies) > S.N_DIES:
                raise EnvError("too many dies")
            ids = list(dies)
        else:
            raise EnvError("dies must be an integer count or a non-empty list of die ids")
        self._charge(COSTS[tool] * len(ids))
        rng = np.random.default_rng([self.seed, 2, self.level.level, self._n_measure])
        self._n_measure += 1
        rows = S.measure(self.wafers[wafer_id].truth, tool, ids, rng)
        self.wafers[wafer_id].measurements.append({"tool": tool, "rows": rows})
        return {"wafer_id": wafer_id, "tool": tool, "data": rows}

    def read_docs(self, topic: str = "index") -> dict:
        topic = str(topic).strip().lower().replace(" ", "_")
        topics = sorted(p.stem for p in DOCS_DIR.glob("*.md"))
        if topic not in topics:
            raise EnvError(f"unknown topic {topic!r}; topics: {topics}")
        return {"topic": topic, "text": (DOCS_DIR / f"{topic}.md").read_text()}

    def submit_recipe(self, recipe: dict) -> dict:
        full = self._check_recipe(recipe)
        rng = np.random.default_rng([self.seed, 3, self.level.level])
        lot = S.simulate_dies(full, self._state, rng, n_wafers=LOT_WAFERS)
        passed = in_spec(lot)
        y = float(passed.mean())
        self.final = {
            "submitted": True,
            "recipe": full,
            "lot_yield": y,
            "budget_spent": round(self.spent, 4),
            "score": max(0.0, y - COST_WEIGHT * self.spent),
            "fail_breakdown": _fail_breakdown(lot),
        }
        self._lot = lot
        self.done = True
        return {"lot_yield": round(y, 4), "score": round(self.final["score"], 4), "done": True}

    # ------------------------------------------------------- generic dispatch
    def step(self, action: dict) -> dict:
        """JSON-style dispatch: {"tool": name, "args": {...}} -> result dict."""
        if self.done:
            return {"ok": False, "error": "episode is over", **self.info()}
        if self.actions >= MAX_ACTIONS:
            self.finish()
            return {"ok": False, "error": "action limit reached; episode over", **self.info()}
        self.actions += 1
        name = action.get("tool") if isinstance(action, dict) else None
        args = action.get("args", {}) if isinstance(action, dict) else {}
        if not isinstance(args, dict):
            args = {}
        fn = {
            "run_wafer": lambda: self.run_wafer(args.get("recipe", {})),
            "measure": lambda: self.measure(args.get("wafer_id"), args.get("tool"), args.get("dies")),
            "read_docs": lambda: self.read_docs(args.get("topic", "index")),
            "submit_recipe": lambda: self.submit_recipe(args.get("recipe", {})),
        }.get(name)
        try:
            if fn is None:
                raise EnvError(f"unknown tool {name!r}")
            res = {"ok": True, "result": fn()}
        except EnvError as e:
            res = {"ok": False, "error": str(e)}
        res.update(self.info())
        self.log.append({"action": action, "response": _short(res)})
        if not self.done and self.actions >= MAX_ACTIONS:
            self.finish()
        return res

    def finish(self) -> dict:
        """End the episode. Without a submission the score is 0."""
        if self.final is None:
            self.final = {
                "submitted": False,
                "recipe": None,
                "lot_yield": 0.0,
                "budget_spent": round(self.spent, 4),
                "score": 0.0,
                "fail_breakdown": {},
            }
        self.done = True
        return self.final

    def result(self) -> dict:
        f = self.finish()
        return {
            "level": self.level.level,
            "seed": self.seed,
            "n_wafers": len(self.wafers),
            "actions": self.actions,
            **f,
        }

    # Debug only: never expose to an agent.
    def _hidden_state(self) -> dict:
        return self._state.to_dict()


def _fail_breakdown(lot) -> dict:
    n = lot["vt"].size
    return {
        "vt_low": float((lot["vt"] < SPEC["vt_min_V"]).sum() / n),
        "vt_high": float((lot["vt"] > SPEC["vt_max_V"]).sum() / n),
        "ion_low": float((lot["ion"] * 1e3 < SPEC["ion_min_mA"]).sum() / n),
        "ioff_high": float((lot["ioff"] * 1e12 > SPEC["ioff_max_pA"]).sum() / n),
        "tox_thin": float((lot["tox_nm"] < SPEC["tox_min_nm"]).sum() / n),
    }


def _short(res):
    s = json.dumps(res, default=float)
    return res if len(s) < 4000 else {**{k: v for k, v in res.items() if k != "result"}, "result": "<truncated>"}
