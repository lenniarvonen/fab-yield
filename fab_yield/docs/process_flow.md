# Process flow

The line makes a single NMOS transistor type (W = 10 um) on a p-type
substrate doped N_sub = 1e16 cm^-3 boron. The recipe is executed in order:

1. **Vt-adjust implant** — boron, `implant_dose` [cm^-2] at `implant_energy` [keV].
   Produces a Gaussian profile (see `implant`).
2. **Activation anneal** — `anneal_temp` [degC] for `anneal_time` [min].
   Boron diffuses (see `diffusion`).
3. **Dry gate oxidation** — `ox_temp` [degC] for `ox_time` [min] in dry O2 on
   (100) Si, starting from 1.5 nm native oxide (see `oxidation`). The
   oxidation is ALSO a thermal step: boron keeps diffusing during it.
   Growing oxide consumes 0.44 nm of silicon per nm of oxide grown; boron in
   the consumed silicon is lost into the oxide.
4. **n+ poly gate** patterned at drawn `gate_length` [um].

Then every die is electrically tested at Vdd = 3.3 V (see `device`).

## Knobs

| knob | unit | range | default | main effects |
|------|------|-------|---------|--------------|
| implant_dose   | cm^-2 | 1e11 .. 1e13 | 1e12 | more dose -> higher Vt, lower Ioff, lower Ion |
| implant_energy | keV   | 5 .. 100     | 20   | deeper implant -> less dose inside depletion region -> lower Vt |
| anneal_temp    | degC  | 800 .. 1100  | 950  | hotter -> more diffusion (Arrhenius, very steep) |
| anneal_time    | min   | 1 .. 120     | 30   | longer -> more diffusion (sqrt(Dt) spreading) |
| ox_temp        | degC  | 800 .. 1050  | 900  | sets oxide growth rate AND adds thermal budget |
| ox_time        | min   | 5 .. 180     | 45   | sets oxide thickness AND adds thermal budget |
| gate_length    | um    | 0.25 .. 2.0  | 1.0  | shorter -> more Ion, but Vt roll-off and higher Ioff |

The knobs are coupled. For example, a thicker oxide raises Vt (smaller Cox)
and lowers Ion; a longer oxidation also diffuses the channel boron deeper,
which lowers Vt. The default recipe does NOT meet spec.
