# Joint limit check: placement_1320_release_window/model_3319.pt

1 episode(s), sampled every 8.33 ms, limits x1 (Franka FCI). Worst fraction of limit over all episodes (>1 = violation):

| Quantity | J1 | J2 | J3 | J4 | J5 | J6 | J7 | Verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| position | 1.00 | 0.30 | 0.92 | 0.84 | 0.11 | 0.70 | 0.90 | VIOLATED |
| velocity | 0.67 | 0.42 | 0.65 | 0.37 | 0.27 | 0.34 | 0.19 | ok |
| acceleration | 2.25 | 1.18 | 1.99 | 0.48 | 0.31 | 0.35 | 0.19 | VIOLATED |
| jerk | 0.53 | 0.05 | 0.57 | 0.08 | 0.07 | 0.05 | 0.01 | ok |

Per episode:

| Env | Episode | Seconds | Ended by | Violations |
|---:|---:|---:|---|---|
| 0 | 0 | 7.96 | held_too_long | position (30 samples); acceleration (9 samples) |

**Overall: FAIL**
