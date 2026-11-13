# Spec and scoring

A die passes if ALL hold:

| parameter | requirement |
|-----------|-------------|
| Vt   | 0.50 V <= Vt <= 0.60 V |
| Ion  | >= 3.5 mA |
| Ioff | <= 20 pA |
| t_ox | >= 7.0 nm (gate-oxide reliability at 3.3 V) |

`submit_recipe(recipe)` ends the episode. The recipe is run on a production
lot of 15 fresh wafers (1020 dies) with the same fab calibration and the same
documented variation. Then

    lot_yield = passing dies / 1020
    score     = max(0, lot_yield - 0.005 * budget_spent)

Not submitting scores 0. A recipe centred in the spec window, and placed
where the device is insensitive to the run-to-run and die-to-die variation,
yields best.
