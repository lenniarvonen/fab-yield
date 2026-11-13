# Gate oxidation: Deal-Grove model (dry O2, (100) Si)

Reference: Deal & Grove, J. Appl. Phys. 36, 3770 (1965); constants from
Plummer, Deal & Griffin, *Silicon VLSI Technology*, Table 6-2.

    x^2 + A x = B (t + tau)
    x(t) = (A/2) * ( sqrt(1 + 4 B (t + tau) / A^2) - 1 )
    tau  = (x_i^2 + A x_i) / B          x_i = 1.5 nm native oxide
    A    = B / (B/A)

Rate constants (T in kelvin, k = 8.617333e-5 eV/K, x in um, t in hours):

    B   = 772 um^2/h        * exp(-1.23 eV / kT)          (parabolic)
    B/A = (6.23e6 / 1.68) um/h * exp(-2.00 eV / kT)       (linear, (100) Si)

Example: 900 degC for 60 min gives about 10.7 nm.

The thin-oxide rapid-growth regime is not modelled beyond the native
oxide. The final gate-oxide thickness t_ox sets Cox = eps_ox / t_ox with
eps_ox = 3.9 * 8.854e-14 F/cm.

Silicon consumed: 0.44 * (t_ox - 1.5 nm). The channel profile is evaluated
below the final Si/SiO2 interface, so the top 0.44*(t_ox-1.5nm) of the
as-annealed boron profile is removed.
