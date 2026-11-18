# Joint limit check: 2026-09-24_18-02-20_grasp_v7b_closure/model_460.pt

1 episode(s), sampled every 8.33 ms, limits x1 (Franka FCI). Worst fraction of limit over all episodes (>1 = violation):

| Quantity | J1 | J2 | J3 | J4 | J5 | J6 | J7 | Verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| position | 0.02 | 0.29 | 0.07 | 0.91 | 0.02 | 0.63 | 0.27 | ok |
| velocity | 0.14 | 0.54 | 0.25 | 0.60 | 0.10 | 0.28 | 0.15 | ok |
| acceleration | 0.19 | 2.29 | 0.45 | 2.15 | 0.15 | 1.43 | 0.11 | VIOLATED |
| jerk | 0.03 | 0.25 | 0.04 | 0.37 | 0.03 | 0.33 | 0.00 | ok |

Per episode:

| Env | Episode | Seconds | Ended by | Violations |
|---:|---:|---:|---|---|
| 0 | 0 | 1.19 | extracted | acceleration (17 samples) |

**Overall: FAIL**
