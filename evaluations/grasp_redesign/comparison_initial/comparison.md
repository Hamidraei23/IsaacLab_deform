| Recording | Success | Seconds | Joint speed p95 (rad/s) | Acceleration p95 (rad/s²) | Jerk p95 (rad/s³) | Hand speed at first hold (m/s) |
|---|---:|---:|---:|---:|---:|---:|
| baseline_900 | 100.0% | 0.97 | 0.893 | 7.95 | 311.6 | 0.334 |
| baseline_920 | 100.0% | 1.05 | 0.873 | 7.92 | 303.2 | 0.310 |
| filtered_900 | 100.0% | 1.97 | 0.534 | 4.18 | 138.9 | 0.229 |

Percentiles pool absolute measurements over valid steps and seven arm joints. Jerk is the control-step difference of recorded acceleration; episode reset boundaries are excluded. First-hold speed is the median over observed hold onsets. These are simulator metrics, not physical hardware limits.

Rewards changed between recordings, so raw episode returns are not comparable. All sheet poses are fixed; target poses and robot resets follow the recorded seeds. The plot uses environment 0, episode 0 as an example, not an aggregate trajectory.
