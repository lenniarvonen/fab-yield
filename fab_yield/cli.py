"""JSON-lines tool server for any agent harness.

    python -m fab_yield.cli --level 3 --seed 0

First line printed: {"task": <prompt>, "tools": [...]}.  Then read one JSON
action per line, {"tool": "...", "args": {...}}, and print one JSON result
per line.  On EOF or episode end, a final {"final": {...}} line is printed.
"""

import argparse
import json
import sys

from .env import FabEnv
from .tools import TOOLS


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--level", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--log", help="write the full trajectory JSON here at the end")
    a = ap.parse_args(argv)
    env = FabEnv(a.level, a.seed)
    print(json.dumps({"task": env.task_prompt(), "tools": TOOLS}), flush=True)
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            action = json.loads(line)
        except json.JSONDecodeError as e:
            print(json.dumps({"ok": False, "error": f"bad JSON: {e}"}), flush=True)
            continue
        print(json.dumps(env.step(action), default=float), flush=True)
        if env.done:
            break
    final = env.result()
    print(json.dumps({"final": final}, default=float), flush=True)
    if a.log:
        with open(a.log, "w") as f:
            json.dump({"level": a.level, "seed": a.seed, "log": env.log, "final": final}, f, indent=1, default=float)


if __name__ == "__main__":
    main()
