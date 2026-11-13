# Boron implant

As-implanted profile is a Gaussian (reflected at the surface with an image
source so the full dose stays in silicon):

    N_impl(x) = Q / (sqrt(2 pi) sigma) * [ exp(-(x-Rp)^2 / 2 sigma^2) + exp(-(x+Rp)^2 / 2 sigma^2) ]

with sigma = dRp before annealing, Q = dose [cm^-2]. Total acceptor
concentration N(x) = N_sub + N_impl(x), N_sub = 1e16 cm^-3.

Projected range Rp and straggle dRp of boron in Si (Gibbons, Johnson &
Mylroie, *Projected Range Statistics*); linear interpolation between rows,
clamped outside the table:

| energy (keV) | Rp (nm) | dRp (nm) |
|---:|---:|---:|
| 5   | 17.8  | 10.5 |
| 10  | 33.3  | 17.1 |
| 20  | 66.2  | 28.3 |
| 30  | 98.7  | 37.3 |
| 40  | 130.2 | 44.3 |
| 60  | 190.3 | 55.6 |
| 80  | 246.5 | 64.1 |
| 100 | 298.8 | 71.0 |
