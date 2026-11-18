# Joint limit check: 2026-09-24_21-17-54_placement_from_v7b_closure/model_880.pt

1 episode(s), sampled every 8.33 ms, limits x1 (Franka FCI). Worst fraction of limit over all episodes (>1 = violation):

| Quantity | J1 | J2 | J3 | J4 | J5 | J6 | J7 | Verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| position | 0.26 | 0.71 | 0.51 | 0.83 | 0.15 | 0.61 | 0.28 | ok |
| velocity | 0.39 | 0.62 | 0.53 | 0.48 | 0.23 | 0.40 | 0.19 | ok |
| acceleration | 0.16 | 0.65 | 0.37 | 0.51 | 0.15 | 0.23 | 0.19 | ok |
| jerk | 0.01 | 0.02 | 0.02 | 0.02 | 0.01 | 0.01 | 0.01 | ok |

Per episode:

| Env | Episode | Seconds | Ended by | Violations |
|---:|---:|---:|---|---|
| 0 | 0 | 2.09 | time_out | none |

**Overall: PASS**
