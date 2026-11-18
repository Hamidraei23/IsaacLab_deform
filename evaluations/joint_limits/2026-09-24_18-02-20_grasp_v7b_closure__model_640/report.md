# Joint limit check: 2026-09-24_18-02-20_grasp_v7b_closure/model_640.pt

1 episode(s), sampled every 8.33 ms, limits x1 (Franka FCI). Worst fraction of limit over all episodes (>1 = violation):

| Quantity | J1 | J2 | J3 | J4 | J5 | J6 | J7 | Verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| position | 0.08 | 0.48 | 0.19 | 0.90 | 0.05 | 0.61 | 0.27 | ok |
| velocity | 0.17 | 0.41 | 0.25 | 0.58 | 0.10 | 0.31 | 0.19 | ok |
| acceleration | 0.25 | 0.75 | 0.45 | 0.59 | 0.13 | 0.43 | 0.15 | ok |
| jerk | 0.03 | 0.04 | 0.04 | 0.10 | 0.02 | 0.10 | 0.01 | ok |

Per episode:

| Env | Episode | Seconds | Ended by | Violations |
|---:|---:|---:|---|---|
| 0 | 0 | 1.56 | extracted | none |

**Overall: PASS**
