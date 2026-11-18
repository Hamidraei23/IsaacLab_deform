# Joint limit check: 2026-09-24_18-02-20_grasp_v7b_closure/model_560.pt

1 episode(s), sampled every 8.33 ms, limits x1 (Franka FCI). Worst fraction of limit over all episodes (>1 = violation):

| Quantity | J1 | J2 | J3 | J4 | J5 | J6 | J7 | Verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| position | 0.03 | 0.36 | 0.10 | 0.91 | 0.04 | 0.61 | 0.27 | ok |
| velocity | 0.12 | 0.50 | 0.21 | 0.59 | 0.11 | 0.29 | 0.19 | ok |
| acceleration | 0.15 | 1.02 | 0.43 | 0.56 | 0.11 | 0.16 | 0.13 | VIOLATED |
| jerk | 0.03 | 0.02 | 0.04 | 0.02 | 0.01 | 0.03 | 0.01 | ok |

Per episode:

| Env | Episode | Seconds | Ended by | Violations |
|---:|---:|---:|---|---|
| 0 | 0 | 1.39 | extracted | acceleration (3 samples) |

**Overall: FAIL**
