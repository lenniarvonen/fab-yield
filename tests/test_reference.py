from fab_yield.env import FabEnv
from fab_yield.reference import run_reference


def test_reference_solves_level1():
    r = run_reference(FabEnv(1, 0))
    assert r["lot_yield"] > 0.95 and r["score"] > 0.95
