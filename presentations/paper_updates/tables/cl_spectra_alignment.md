# Continual-phase spectra and singular-vector alignment (FLAME-CL, first_appearance)

Median over the 5 experts; each cell is **fc2 / fc1**. Seed 42 for every setup; Setup 4 also over 4 seeds below. r90 = rank at 90% cumulative energy (ambient 128). Kept = data-aware energy in the top-32 singular directions. A(r) = ||V_r^T Q_r||_F^2 / r (random subspaces: r/128).

| Setup | Stage | tasks | r90 input | r90 weight | r90 data-aware | kept@32 data-aware | A(8) | A(16) | A(32) |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 1 | PHENO | 4 / 4 | 65 / 66 | 37 / 45 | 0.872 / 0.844 | 0.17 / 0.13 | 0.21 / 0.20 | 0.30 / 0.31 |
| 1 | 2 | LOS | 5 / 3 | 9 / 65 | 2 / 74 | 1.000 / 0.511 | 0.41 / 0.11 | 0.46 / 0.14 | 0.46 / 0.27 |
| 1 | 3 | IHM | 4 / 2 | 9 / 65 | 3 / 69 | 1.000 / 0.646 | 0.34 / 0.10 | 0.44 / 0.15 | 0.45 / 0.26 |
| 2 | 1 | MOR | 3 / 7 | 66 / 66 | 47 / 50 | 0.826 / 0.782 | 0.13 / 0.12 | 0.17 / 0.17 | 0.28 / 0.28 |
| 2 | 2 | RAD | 8 / 7 | 10 / 66 | 3 / 62 | 1.000 / 0.643 | 0.35 / 0.09 | 0.36 / 0.15 | 0.43 / 0.27 |
| 3 | 1 | DENSITY | 6 / 12 | 66 / 65 | 50 / 55 | 0.772 / 0.736 | 0.10 / 0.11 | 0.16 / 0.15 | 0.27 / 0.26 |
| 3 | 2 | BIRADS | 5 / 6 | 9 / 66 | 6 / 62 | 1.000 / 0.691 | 0.41 / 0.11 | 0.41 / 0.16 | 0.43 / 0.27 |
| 3 | 3 | RISK | 2 / 2 | 7 / 66 | 2 / 69 | 1.000 / 0.559 | 0.36 / 0.06 | 0.32 / 0.12 | 0.39 / 0.25 |
| 4 | 1 | PHENO+DENSITY | 4 / 4 | 66 / 66 | 41 / 41 | 0.860 / 0.852 | 0.18 / 0.19 | 0.21 / 0.23 | 0.31 / 0.32 |
| 4 | 2 | LOS+BIRADS+MOR | 7 / 5 | 12 / 66 | 1 / 68 | 1.000 / 0.686 | 0.43 / 0.14 | 0.44 / 0.16 | 0.50 / 0.27 |
| 4 | 3 | IHM+RISK+RAD | 4 / 5 | 11 / 65 | 3 / 61 | 1.000 / 0.763 | 0.35 / 0.13 | 0.41 / 0.16 | 0.47 / 0.27 |

## Setup 4 across seeds (42, 703, 8622, 92784)

Median over 5 experts x 4 seeds, [min, max] over the 4 per-seed medians.

| Stage | sublayer | r90 data-aware | kept@32 data-aware | A(16) |
|---|---|---|---|---|
| 1 | fc2 | 41 [39, 42] | 0.855 [0.841, 0.860] | 0.22 [0.21, 0.23] |
| 1 | fc1 | 43 [41, 45] | 0.835 [0.819, 0.852] | 0.22 [0.21, 0.23] |
| 2 | fc2 | 2 [1, 4] | 1.000 [0.999, 1.000] | 0.44 [0.41, 0.44] |
| 2 | fc1 | 68 [65, 69] | 0.676 [0.651, 0.697] | 0.16 [0.15, 0.16] |
| 3 | fc2 | 3 [3, 3] | 1.000 [0.999, 1.000] | 0.39 [0.39, 0.41] |
| 3 | fc1 | 62 [57, 71] | 0.734 [0.604, 0.763] | 0.16 [0.15, 0.16] |

## Pooled over all setups (seed 42, per sublayer: median [range])

- zero-init fc2 (stages>=2): r90 data-aware: 3 [1, 8]
- zero-init fc2 (stages>=2): kept@32 data-aware: 1.000 [min 0.998]
- zero-init fc2 (stages>=2): A(16): 0.41 [0.22, 0.51]
- random-init fc1 (stages>=2): r90 data-aware: 66 [42, 86]
- random-init fc1 (stages>=2): kept@32 data-aware: 0.646 [min 0.400]
- random-init fc1 (stages>=2): A(16): 0.16 [0.12, 0.19]
- stage 1, both sublayers: r90 data-aware: 46 [23, 59]
- stage 1, both sublayers: kept@32 data-aware: 0.817
- stage 1, both sublayers: A(16): 0.18
- all sublayers: r90 input: 4 [1, 22]
