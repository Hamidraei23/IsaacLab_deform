# Joint limit check: 2026-09-24_18-02-20_grasp_v7b_closure/model_280.pt

1 episode(s), sampled every 8.33 ms, limits x1 (Franka FCI). Worst fraction of limit over all episodes (>1 = violation):

| Quantity | J1 | J2 | J3 | J4 | J5 | J6 | J7 | Verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| position | 0.03 | 0.29 | 0.08 | 0.91 | 0.01 | 0.61 | 0.27 | ok |
| velocity | 0.18 | 0.60 | 0.35 | 0.61 | 0.07 | 0.13 | 0.09 | ok |
| acceleration | 0.26 | 3.04 | 0.53 | 3.02 | 0.18 | 2.19 | 0.27 | VIOLATED |
| jerk | 0.05 | 0.38 | 0.07 | 0.56 | 0.03 | 0.49 | 0.06 | ok |

Per episode:

| Env | Episode | Seconds | Ended by | Violations |
|---:|---:|---:|---|---|
| 0 | 0 | 1.06 | extracted | acceleration (19 samples) |

**Overall: FAIL**
