"""One-command evaluation: python -m fab_yield.eval

Runs every policy on every level and seed, prints a table and writes
results/results.json.  Deterministic: same seeds -> identical numbers.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from .baselines import do_nothing, open_loop, stop_early
from .env import FabEnv
from .reference import run_reference

POLICIES = {
    "do_nothing": do_nothing,
    "stop_early": stop_early,
    "open_loop_physics": open_loop,
    "reference": run_reference,
}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--levels", type=int, nargs="+", default=[1, 2, 3])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--policies", nargs="+", default=list(POLICIES), choices=list(POLICIES))
    ap.add_argument("--out", default="results/results.json")
    args = ap.parse_args(argv)

    rows = []
    for level in args.levels:
        for pol in args.policies:
            for seed in args.seeds:
                t0 = time.time()
                r = POLICIES[pol](FabEnv(level, seed))
                r = {"policy": pol, **r, "seconds": round(time.time() - t0, 2)}
                rows.append(r)
                print(f"L{level} {pol:18s} seed={seed}  yield={r['lot_yield']:.3f}  score={r['score']:.3f}  "
                      f"spent={r['budget_spent']:.2f}  ({r['seconds']}s)", flush=True)

    print("\n| level | policy | mean score | min | max | mean yield | mean budget |")
    print("|---|---|---|---|---|---|---|")
    summary = []
    for level in args.levels:
        for pol in args.policies:
            sel = [r for r in rows if r["level"] == level and r["policy"] == pol]
            sc = np.array([r["score"] for r in sel])
            y = np.array([r["lot_yield"] for r in sel])
            b = np.array([r["budget_spent"] for r in sel])
            summary.append({"level": level, "policy": pol, "mean_score": float(sc.mean()),
                            "min_score": float(sc.min()), "max_score": float(sc.max()),
                            "mean_yield": float(y.mean()), "mean_budget": float(b.mean())})
            print(f"| L{level} | {pol} | {sc.mean():.3f} | {sc.min():.3f} | {sc.max():.3f} | {y.mean():.3f} | {b.mean():.2f} |")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"summary": summary, "runs": rows}, indent=2, default=float))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
