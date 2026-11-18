# Joint limit check: 2026-09-24_18-02-20_grasp_v7b_closure/model_600.pt

1 episode(s), sampled every 8.33 ms, limits x1 (Franka FCI). Worst fraction of limit over all episodes (>1 = violation):

| Quantity | J1 | J2 | J3 | J4 | J5 | J6 | J7 | Verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| position | 0.05 | 0.41 | 0.18 | 0.90 | 0.05 | 0.61 | 0.27 | ok |
| velocity | 0.12 | 0.46 | 0.28 | 0.58 | 0.10 | 0.24 | 0.19 | ok |
| acceleration | 0.23 | 0.84 | 0.51 | 0.57 | 0.13 | 0.35 | 0.14 | ok |
| jerk | 0.08 | 0.04 | 0.10 | 0.12 | 0.01 | 0.11 | 0.01 | ok |

Per episode:

| Env | Episode | Seconds | Ended by | Violations |
|---:|---:|---:|---|---|
| 0 | 0 | 1.49 | extracted | none |

**Overall: PASS**
