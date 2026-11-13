# Metrology

`measure(wafer_id, tool, dies)` — `dies` is a list of die ids, or an integer
n (n dies spread evenly over the id range). Default 5. Cost is per die/site.

| tool | returns per die | noise (1 sigma) | cost per site |
|------|-----------------|-----------------|---------------|
| `ellipsometer` | `tox_nm` gate-oxide thickness | 0.15 nm | 0.005 |
| `four_point_probe` | `sheet_res_ohm_sq` of the boron layer (monitor wafer) | 1 % | 0.005 |
| `electrical` | `vt_V`, `ion_mA`, `ioff_pA` | Vt 3 mV, Ion 0.5 %, Ioff 0.03 decades | 0.02 |

Measurement noise is independent on every call, so re-measuring averages it
down. Die-to-die and run-to-run process variation (see `variation`) are real
differences between dies and wafers, not measurement noise: measuring more
dies tells you about the spread you will see in production.

`run_wafer(recipe)` costs 1.0 and returns a wafer id; it gives no data by
itself.
