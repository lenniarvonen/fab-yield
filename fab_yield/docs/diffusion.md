# Boron diffusion and thermal budget

Intrinsic boron diffusivity in Si (Plummer, Deal & Griffin, Table 7-5):

    D(T) = 0.76 cm^2/s * exp(-3.46 eV / kT)

Examples: D(900 C) ~ 1.0e-15, D(1000 C) ~ 1.5e-14, D(1100 C) ~ 1.5e-13 cm^2/s.
Every +10 degC raises D by roughly 25-30 %.

Both furnace steps contribute to the thermal budget (t in seconds):

    Dt_total = D(T_anneal) * t_anneal + D(T_ox) * t_ox

A Gaussian stays Gaussian under constant-D diffusion, so after all heat
treatments

    sigma^2 = dRp^2 + 2 * Dt_total

with Rp unchanged and the dose conserved (apart from silicon consumed by
oxidation, see `oxidation`). Concentration-dependent (extrinsic) and
transient-enhanced diffusion are neglected.
