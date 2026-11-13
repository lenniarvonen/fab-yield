# Variation and calibration

## Wafer
100 mm wafer, 9 mm square dies, 3 mm edge exclusion -> 68 dies, ids 0..67,
numbered row by row from the top. `measure` returns x_mm, y_mm per die;
r = sqrt(x^2 + y^2) / 50 mm is the normalised radius used below.

## Systematic radial patterns
- Oxide: t_ox(r) = t_ox_center * (1 + g * r^2). Nominal g = 0.02.
- Gate length: L(r) = L_drawn + 0.02 um * r^2.

## Run-to-run (whole wafer, every run_wafer call)
- Each furnace's real temperature: +/- 1.0 degC (1 sigma)
- Implanter dose: +/- 1.0 % (1 sigma)

## Die-to-die (independent per die)
- Dose: 1.5 %, oxide thickness: 1.0 %, gate length: 0.02 um,
  fixed oxide charge: 1e10 q/cm^2 (all 1 sigma).

Level 1 uses half of every run-to-run and die-to-die sigma above.

## Calibration (level 3)
In level 3 the tools carry constant, unknown calibration errors (the same
for every wafer you run and for the production lot):
- anneal furnace: real T = setpoint + offset, |offset| up to 20 degC
- oxidation furnace: real T = setpoint + offset, |offset| up to 12 degC
- implanter: delivered dose = setpoint * factor, factor in 0.90 .. 1.10
- oxide radial coefficient g in 0 .. 0.04 (instead of 0.02)

In levels 1 and 2 all tools are calibrated (offsets 0, factor 1, g = 0.02).
