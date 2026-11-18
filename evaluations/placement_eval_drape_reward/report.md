| Checkpoint | Extracted | Released (≤6 cm) | High releases / ep | Aborted | Success | Best coverage | Centre progress | Seconds | Return |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 18-51-43_placement_v7b@800 | 91% | 0% | 0.09 | 31% | 0% | 0.00 | 0.38 | 11.5 | -432 |
| 19-17-18_placement_v7b@1060 | 78% | 0% | 0.00 | 78% | 0% | 0.01 | 0.57 | 9.1 | 744 |
| 20-05-04_placement_v7b@1340 | 100% | 0% | 0.00 | 3% | 0% | 0.26 | 0.82 | 32.6 | -5,921 |
| 21-17-54_placement_v7b@820 | 91% | 0% | 0.00 | 41% | 0% | 0.03 | 0.41 | 20.0 | -3,161 |
| 21-36-39_placement_wandb@920 | 91% | 0% | 0.16 | 19% | 0% | 0.01 | 0.37 | 30.1 | -7,762 |
| placement_1320_release_window@3319 | 97% | 0% | 0.00 | 0% | 0% | 0.52 | 0.85 | 33.3 | 2,462 |
| placement_3319_joint_limits@3340 | 100% | 0% | 0.00 | 0% | 0% | 0.46 | 0.81 | 33.3 | 2,681 |
| placement_880_open_intent@920 | 88% | 0% | 0.00 | 22% | 0% | 0.04 | 0.37 | 29.3 | -7,465 |
| placement_880_open_intent_g1@920 | 94% | 0% | 0.00 | 12% | 0% | 0.07 | 0.43 | 31.6 | -7,030 |
| placement_880_open_intent_g2@1440 | 94% | 0% | 0.00 | 0% | 0% | 0.02 | 0.26 | 28.2 | 4,521 |
| 18-02-20_grasp_v7b_closure@720 | 97% | 0% | 0.22 | 6% | 0% | 0.00 | 0.01 | 8.7 | 1,639 |

Endings:

- **18-51-43_placement_v7b@800**: deformable_out_of_bounds 13, released_high 10, time_out 4, ee_below_table 4, joint_vel_out_of_limit 1
- **19-17-18_placement_v7b@1060**: released_high 25, time_out 7
- **20-05-04_placement_v7b@1340**: time_out 30, ee_below_table 1, released_high 1
- **21-17-54_placement_v7b@820**: released_high 13, deformable_out_of_bounds 12, time_out 7
- **21-36-39_placement_wandb@920**: time_out 25, released_high 6, ee_below_table 1
- **placement_1320_release_window@3319**: time_out 32
- **placement_3319_joint_limits@3340**: time_out 32
- **placement_880_open_intent@920**: time_out 25, released_high 6, time_out,released_high 1
- **placement_880_open_intent_g1@920**: time_out 27, released_high 4, ee_below_table 1
- **placement_880_open_intent_g2@1440**: time_out 23, ee_below_table 8, deformable_out_of_bounds 1
- **18-02-20_grasp_v7b_closure@720**: deformable_out_of_bounds 22, ee_below_table 4, time_out 4, released_high 2

Mean reward per episode by term:

| Checkpoint | ee_speed | grasp_progress | grasp_stage | gripper_closures | lift_milestones | motion_cost | placement | table_clearance |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 18-51-43_placement_v7b@800 | 0 | 97 | 191 | -23 | 5,133 | -4,627 | -1,203 | -0 |
| 19-17-18_placement_v7b@1060 | 0 | 92 | 165 | -97 | 4,693 | -2,602 | -1,509 | 0 |
| 20-05-04_placement_v7b@1340 | 0 | 101 | 210 | -3 | 5,386 | -13,730 | 2,114 | -0 |
| 21-17-54_placement_v7b@820 | 0 | 89 | 191 | -16 | 5,197 | -7,361 | -1,262 | -0 |
| 21-36-39_placement_wandb@920 | 0 | 95 | 191 | -23 | 5,000 | -12,020 | -1,005 | -0 |
| placement_1320_release_window@3319 | 0 | 111 | 204 | 0 | 5,381 | -8,100 | 4,871 | -4 |
| placement_3319_joint_limits@3340 | 0 | 100 | 210 | -5 | 5,400 | -7,744 | 4,725 | -5 |
| placement_880_open_intent@920 | 0 | 94 | 184 | -67 | 4,718 | -11,663 | -731 | -0 |
| placement_880_open_intent_g1@920 | 0 | 98 | 197 | -33 | 5,070 | -12,144 | -218 | -0 |
| placement_880_open_intent_g2@1440 | 0 | 96 | 197 | -2 | 5,272 | -245 | -798 | -0 |
| 18-02-20_grasp_v7b_closure@720 | 0 | 101 | 204 | -2 | 5,234 | -2,606 | -1,290 | -1 |

Placement parts per episode:

| Checkpoint | abort | approach | blocked | coverage | failure | goal_proximity | high_release | hold_over_target | open_intent | release | retreat | success |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 18-51-43_placement_v7b@800 | -469 | 337 | 0 | 0 | -1,099 | 2 | -90 | 0 | 0 | 0 | 116 | 0 |
| 19-17-18_placement_v7b@1060 | -1,172 | 701 | 0 | 1 | -1,044 | 5 | 0 | 0 | 0 | 0 | 0 | 0 |
| 20-05-04_placement_v7b@1340 | -47 | 1,130 | 0 | 2,103 | -1,016 | 126 | 0 | -182 | 0 | 0 | 1 | 0 |
| 21-17-54_placement_v7b@820 | -609 | 405 | 0 | 10 | -1,075 | 5 | 0 | 0 | 0 | 0 | 3 | 0 |
| 21-36-39_placement_wandb@920 | -281 | 320 | 0 | 109 | -1,088 | 44 | -113 | 0 | 0 | 0 | 5 | 0 |
| placement_1320_release_window@3319 | 0 | 1,246 | -8,498 | 15,614 | -597 | 914 | 0 | -3,809 | 0 | 0 | 0 | 0 |
| placement_3319_joint_limits@3340 | 0 | 1,111 | -5,296 | 12,877 | -682 | 1,028 | 0 | -4,313 | 0 | 0 | 0 | 0 |
| placement_880_open_intent@920 | -328 | 336 | 0 | 257 | -1,057 | 53 | 0 | 0 | 0 | 0 | 8 | 0 |
| placement_880_open_intent_g1@920 | -188 | 407 | 0 | 589 | -1,028 | 1 | 0 | 0 | 0 | 0 | 0 | 0 |
| placement_880_open_intent_g2@1440 | 0 | 219 | 0 | 61 | -1,078 | 1 | 0 | 0 | 0 | 0 | 0 | 0 |
| 18-02-20_grasp_v7b_closure@720 | -94 | 3 | 0 | 0 | -1,100 | 0 | -102 | 0 | 0 | 0 | 3 | 0 |
