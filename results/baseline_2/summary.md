# SABLE classical baseline

total episodes: 1688

## Block A -- classical scaling (comms OFF, 25x25, d=0.30)

| policy / n | runs | compl. | avg ticks (done) | ticks capped | final cov | redundancy | conflicts | bytes deliv/drop |
|---|---|---|---|---|---|---|---|---|
| move_toward_frontier_astar n=1 | 8 | 8/8 (100%) | 434.4 | 434.4 | 1.000 | 0.000 | 0.00 | 0/0 |
| move_toward_frontier_astar n=2 | 8 | 2/8 (25%) | 388.0 | 2347.0 | 0.622 | 0.764 | 4289.00 | 0/0 |
| move_toward_frontier_astar n=4 | 18 | 4/18 (22%) | 402.5 | 2422.8 | 0.516 | 0.694 | 9439.56 | 0/0 |
| move_toward_frontier_astar n=8 | 8 | 0/8 (0%) | nan | 3000.0 | 0.588 | 0.787 | 23738.50 | 0/0 |
| move_toward_unknown_bfs n=1 | 8 | 8/8 (100%) | 439.0 | 439.0 | 1.000 | 0.000 | 0.00 | 0/0 |
| move_toward_unknown_bfs n=2 | 8 | 3/8 (38%) | 395.0 | 2023.1 | 0.626 | 0.634 | 3637.75 | 0/0 |
| move_toward_unknown_bfs n=4 | 8 | 1/8 (12%) | 433.0 | 2679.1 | 0.615 | 0.695 | 10390.62 | 0/0 |
| move_toward_unknown_bfs n=8 | 8 | 0/8 (0%) | nan | 3000.0 | 0.820 | 0.883 | 23518.25 | 0/0 |
| coordinated_frontier_greedy n=1 | 8 | 8/8 (100%) | 433.9 | 433.9 | 1.000 | 0.000 | 0.00 | 0/0 |
| coordinated_frontier_greedy n=2 | 8 | 4/8 (50%) | 405.0 | 1702.5 | 0.793 | 0.899 | 2819.50 | 0/0 |
| coordinated_frontier_greedy n=4 | 18 | 9/18 (50%) | 382.1 | 1691.1 | 0.812 | 1.532 | 5980.56 | 0/0 |
| coordinated_frontier_greedy n=8 | 8 | 5/8 (62%) | 292.4 | 1307.8 | 0.974 | 3.059 | 9060.75 | 0/0 |
| coordinated_frontier_hungarian n=1 | 8 | 8/8 (100%) | 433.9 | 433.9 | 1.000 | 0.000 | 0.00 | 0/0 |
| coordinated_frontier_hungarian n=2 | 8 | 7/8 (88%) | 389.9 | 716.1 | 0.980 | 0.905 | 664.25 | 0/0 |
| coordinated_frontier_hungarian n=4 | 18 | 16/18 (89%) | 249.6 | 555.2 | 0.996 | 1.647 | 1226.89 | 0/0 |
| coordinated_frontier_hungarian n=8 | 8 | 8/8 (100%) | 141.5 | 141.5 | 1.000 | 1.861 | 49.50 | 0/0 |

## Block B -- comms ladder (25x25, n=4, d=0.30)

| policy / comms | runs | compl. | avg ticks (done) | ticks capped | final cov | redundancy | conflicts | bytes deliv/drop |
|---|---|---|---|---|---|---|---|---|
| move_toward_frontier_astar [off] | 18 | 4/18 (22%) | 402.5 | 2422.8 | 0.516 | 0.694 | 9439.56 | 0/0 |
| move_toward_frontier_astar [lossless] | 10 | 3/10 (30%) | 215.7 | 2164.7 | 0.942 | 0.473 | 8214.10 | 9218/0 |
| move_toward_frontier_astar [drop0.3] | 10 | 1/10 (10%) | 178.0 | 2717.8 | 0.613 | 0.633 | 10560.70 | 5287/2294 |
| move_toward_frontier_astar [drop0.7] | 10 | 3/10 (30%) | 356.7 | 2207.0 | 0.648 | 0.581 | 8484.20 | 2728/6526 |
| coordinated_frontier_greedy [off] | 18 | 9/18 (50%) | 382.1 | 1691.1 | 0.812 | 1.532 | 5980.56 | 0/0 |
| coordinated_frontier_greedy [lossless] | 10 | 5/10 (50%) | 148.4 | 1574.2 | 0.960 | 0.375 | 5871.00 | 9425/0 |
| coordinated_frontier_greedy [drop0.3] | 10 | 7/10 (70%) | 271.6 | 1090.1 | 0.953 | 1.037 | 3716.10 | 9010/3820 |
| coordinated_frontier_greedy [drop0.7] | 10 | 7/10 (70%) | 306.6 | 1114.6 | 0.936 | 1.399 | 3653.60 | 5317/12650 |
| coordinated_frontier_hungarian [off] | 18 | 16/18 (89%) | 249.6 | 555.2 | 0.996 | 1.647 | 1226.89 | 0/0 |
| coordinated_frontier_hungarian [lossless] | 10 | 8/10 (80%) | 120.5 | 696.4 | 0.999 | 0.482 | 2314.60 | 9873/0 |
| coordinated_frontier_hungarian [drop0.3] | 10 | 9/10 (90%) | 243.4 | 519.1 | 1.000 | 1.242 | 1337.80 | 9907/4229 |
| coordinated_frontier_hungarian [drop0.7] | 10 | 8/10 (80%) | 299.0 | 839.2 | 0.938 | 1.495 | 2524.40 | 5531/13025 |

## Block C -- structured maps (10x10), per map (comms OFF)

| map (all policies/n, comms off) | runs | compl. | avg ticks (done) | ticks capped | final cov | redundancy | conflicts | bytes deliv/drop |
|---|---|---|---|---|---|---|---|---|
| empty | 40 | 26/40 (65%) | 49.7 | 557.3 | 0.888 | 0.691 | 1568.72 | 0/0 |
| bordered_room | 40 | 24/40 (60%) | 31.6 | 619.0 | 0.884 | 0.712 | 1714.05 | 0/0 |
| single_corridor | 40 | 30/40 (75%) | 3.3 | 377.5 | 0.945 | 0.392 | 1048.85 | 0/0 |
| bottleneck | 40 | 27/40 (68%) | 50.3 | 521.5 | 0.875 | 0.840 | 1341.22 | 0/0 |
| two_bottlenecks | 40 | 27/40 (68%) | 48.3 | 520.1 | 0.915 | 0.684 | 1415.78 | 0/0 |
| four_rooms | 40 | 27/40 (68%) | 56.2 | 525.5 | 0.929 | 0.921 | 1484.72 | 0/0 |
| nested_rooms | 40 | 20/40 (50%) | 47.4 | 773.7 | 0.761 | 0.756 | 2161.50 | 0/0 |
| comb | 40 | 10/40 (25%) | 40.3 | 1135.1 | 0.733 | 0.517 | 3270.15 | 0/0 |
| u_trap | 40 | 20/40 (50%) | 48.5 | 774.3 | 0.861 | 0.814 | 2295.80 | 0/0 |
| central_block | 40 | 19/40 (48%) | 47.3 | 810.0 | 0.793 | 0.916 | 2229.78 | 0/0 |
| pillars | 40 | 25/40 (62%) | 46.4 | 591.5 | 0.882 | 0.896 | 1561.15 | 0/0 |
| dense_pillars | 40 | 23/40 (57%) | 59.3 | 671.6 | 0.886 | 0.941 | 2009.03 | 0/0 |
| zigzag | 40 | 26/40 (65%) | 37.6 | 549.5 | 0.887 | 0.515 | 1417.15 | 0/0 |
| spiral | 40 | 14/40 (35%) | 39.8 | 988.9 | 0.711 | 0.641 | 2828.45 | 0/0 |
| ring | 40 | 23/40 (57%) | 22.1 | 650.2 | 0.886 | 0.589 | 1786.90 | 0/0 |
| cross_corridors | 40 | 22/40 (55%) | 18.0 | 684.9 | 0.845 | 0.504 | 1942.42 | 0/0 |
| diagonal_wall | 40 | 23/40 (57%) | 51.7 | 667.2 | 0.823 | 0.691 | 2010.75 | 0/0 |
| maze | 40 | 19/40 (48%) | 45.3 | 809.0 | 0.790 | 0.692 | 2385.95 | 0/0 |

## Block C -- comms OFF vs lossless (10x10, aggregated over maps)

| policy / n / comms | runs | compl. | avg ticks (done) | ticks capped | final cov | redundancy | conflicts | bytes deliv/drop |
|---|---|---|---|---|---|---|---|---|
| move_toward_frontier_astar n=2 [off] | 90 | 18/90 (20%) | 56.7 | 1211.3 | 0.670 | 0.498 | 2364.80 | 0/0 |
| move_toward_frontier_astar n=2 [lossless] | 90 | 61/90 (68%) | 36.6 | 508.2 | 0.868 | 0.327 | 956.49 | 481/0 |
| move_toward_frontier_astar n=4 [off] | 90 | 11/90 (12%) | 37.5 | 1321.2 | 0.715 | 0.592 | 5237.07 | 0/0 |
| move_toward_frontier_astar n=4 [lossless] | 90 | 67/90 (74%) | 27.3 | 403.6 | 0.949 | 0.684 | 1539.59 | 1602/0 |
| move_toward_unknown_bfs n=2 [off] | 90 | 25/90 (28%) | 45.1 | 1095.9 | 0.713 | 0.451 | 2135.00 | 0/0 |
| move_toward_unknown_bfs n=2 [lossless] | 90 | 70/90 (78%) | 37.2 | 362.3 | 0.915 | 0.313 | 658.96 | 510/0 |
| move_toward_unknown_bfs n=4 [off] | 90 | 32/90 (36%) | 29.0 | 977.0 | 0.828 | 0.664 | 3848.11 | 0/0 |
| move_toward_unknown_bfs n=4 [lossless] | 90 | 69/90 (77%) | 22.0 | 366.9 | 0.960 | 0.527 | 1401.23 | 1619/0 |
| coordinated_frontier_greedy n=2 [off] | 90 | 68/90 (76%) | 54.2 | 407.7 | 0.913 | 0.682 | 720.47 | 0/0 |
| coordinated_frontier_greedy n=2 [lossless] | 90 | 78/90 (87%) | 33.9 | 229.4 | 0.966 | 0.291 | 392.80 | 542/0 |
| coordinated_frontier_greedy n=4 [off] | 90 | 84/90 (93%) | 44.0 | 141.1 | 0.985 | 1.478 | 422.72 | 0/0 |
| coordinated_frontier_greedy n=4 [lossless] | 90 | 83/90 (92%) | 20.7 | 135.7 | 0.989 | 0.660 | 465.89 | 1710/0 |
| coordinated_frontier_hungarian n=2 [off] | 90 | 82/90 (91%) | 43.0 | 172.5 | 0.979 | 0.460 | 260.71 | 0/0 |
| coordinated_frontier_hungarian n=2 [lossless] | 90 | 85/90 (94%) | 34.1 | 115.5 | 0.996 | 0.284 | 162.89 | 562/0 |
| coordinated_frontier_hungarian n=4 [off] | 90 | 85/90 (94%) | 25.4 | 107.3 | 0.995 | 0.826 | 332.18 | 0/0 |
| coordinated_frontier_hungarian n=4 [lossless] | 90 | 86/90 (96%) | 18.7 | 84.5 | 0.998 | 0.583 | 266.29 | 1728/0 |

