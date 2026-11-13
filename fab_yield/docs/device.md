# Device model (electrical test at 300 K)

Constants: q = 1.602e-19 C, kT/q = 25.85 mV, n_i = 1e10 cm^-3,
eps_si = 11.7 eps0, eps_ox = 3.9 eps0, eps0 = 8.854e-14 F/cm,
W = 10 um, Vdd = 3.3 V, fixed oxide charge Qf = 5e10 q/cm^2.

## Threshold voltage (long channel)

    phi_F  = (kT/q) ln(N_sub / n_i)                   (= 0.357 V)
    V_FB   = -(0.56 + phi_F) - q Qf / Cox             (n+ poly gate)
    Vt_long = V_FB + 2 phi_F + Q_dep / Cox

The channel doping N(x) is non-uniform, so the depletion width W_d at the
onset of strong inversion is found from Poisson's equation in the depletion
approximation:

    (q / eps_si) * integral_0^W_d  x N(x) dx = 2 phi_F
    Q_dep = q * integral_0^W_d N(x) dx

(Only the part of the implanted dose that sits inside W_d raises Vt.)
Numerically: 0..1 um grid with 801 points.

## Short-channel Vt roll-off and DIBL (Liu et al., IEEE TED 1993)

    l_t   = sqrt(eps_si * t_ox * W_d / eps_ox)
    dVt   = [2 (V_bi - 2 phi_F) + Vdd] * [exp(-L / 2 l_t) + 2 exp(-L / l_t)]
    Vt    = Vt_long - dVt,     V_bi = 0.9 V

Short gates and thick oxides / light doping (long l_t) lower Vt.

## Off current (Vgs = 0, Vds = Vdd)

    S    = ln(10) (kT/q) (1 + C_dep / Cox),  C_dep = eps_si / W_d   [V/decade]
    Ioff = 1e-7 A * (W / L) * 10^(-Vt / S)

(Vt is defined at a drain current of 1e-7 A * W/L.)

## On current (Vgs = Vds = Vdd), velocity saturation (C. Hu, 2010)

    mu_eff = 450 / (1 + 0.12 * (10 nm / t_ox) * (Vdd - Vt))     cm^2/Vs
    E_c    = 2 v_sat / mu_eff,    v_sat = 8e6 cm/s
    Ion    = W Cox v_sat (Vdd - Vt)^2 / ((Vdd - Vt) + E_c L)

## Sheet resistance of the boron layer (monitor wafer)

    Rs = 1 / ( q * integral mu_p(N_impl) N_impl dx )
    mu_p(N) = 44.9 + (470.5 - 44.9) / (1 + (N / 2.23e17)^0.719)   (Caughey-Thomas)

Rs is measured on a co-processed n-type monitor wafer, so only the
implanted boron conducts. It responds to dose (strongly) and to diffusion
(weakly, through mobility).
