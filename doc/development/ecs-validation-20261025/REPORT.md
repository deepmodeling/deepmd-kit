# ECS current-master comparison — measured 2026-10-25, finalized 2026-10-26

BASE: 2ba617bc3db67adee2b8ba00c71e9fa1c2d2bb8c, including merged residual-inplace #6058. Old668-base numbers are not substituted. Candidate is the isolated port of PR#2; its AOT branch-specialization issue remains unresolved.

Recorded 135/135 process results: {'completed': 120, 'oom': 15}. Missing cells are NOT passes. Three processes ×3 frames ×3 accuracy repeats per arm; 8 extra warmups/30 wall+Event samples, separate steady replay. Same physical GPU within each case. TF32/AMP off.

## Three-column PASS/FAIL overview

Strict columns cover all four fields: Torch self-repeat; master self-repeat and versus Torch; candidate self-repeat, versus Torch and versus master. A missing Torch reference is N/A, never PASS. The meeting band is a separately reported empirical diagnostic, not a rigorous error bound or a waiver of strict failure.

| Model/system | Atoms | Compiled Torch | Current master | Candidate |
| --- | ---: | --- | --- | --- |
| Neo | 96 | strict self=PASS | strict self=PASS; vs Torch=PASS; band=PASS (2.171x) | strict self=PASS; vs Torch=PASS; vs master=PASS; band=PASS (2.355x) |
| Neo | 512 | strict self=PASS | strict self=PASS; vs Torch=PASS; band=PASS (2.406x) | strict self=PASS; vs Torch=FAIL; vs master=PASS; band=PASS (2.261x) |
| Neo | 4096 | strict self=N/A | strict self=FAIL; vs Torch=N/A; band=N/A | strict self=FAIL; vs Torch=N/A; vs master=FAIL; band=N/A |
| NeoCu | 96 | strict self=PASS | strict self=PASS; vs Torch=PASS; band=PASS (1.989x) | strict self=PASS; vs Torch=PASS; vs master=PASS; band=PASS (2.084x) |
| NeoCu | 512 | strict self=PASS | strict self=PASS; vs Torch=PASS; band=PASS (2.499x) | strict self=PASS; vs Torch=PASS; vs master=PASS; band=PASS (2.863x) |
| NeoCu | 4096 | strict self=PASS | strict self=PASS; vs Torch=PASS; band=FAIL (6.690x) | strict self=PASS; vs Torch=PASS; vs master=PASS; band=FAIL (6.707x) |
| Nano | 96 | strict self=FAIL | strict self=FAIL; vs Torch=FAIL; band=PASS (1.316x) | strict self=FAIL; vs Torch=FAIL; vs master=FAIL; band=PASS (1.498x) |
| Nano | 512 | strict self=FAIL | strict self=FAIL; vs Torch=FAIL; band=PASS (1.839x) | strict self=FAIL; vs Torch=FAIL; vs master=FAIL; band=PASS (1.831x) |
| Nano | 4096 | strict self=FAIL | strict self=FAIL; vs Torch=FAIL; band=FAIL (3.031x) | strict self=FAIL; vs Torch=FAIL; vs master=FAIL; band=FAIL (3.171x) |
| Mini | 96 | strict self=FAIL | strict self=FAIL; vs Torch=FAIL; band=PASS (1.474x) | strict self=FAIL; vs Torch=FAIL; vs master=FAIL; band=PASS (1.433x) |
| Mini | 512 | strict self=FAIL | strict self=FAIL; vs Torch=FAIL; band=PASS (1.519x) | strict self=FAIL; vs Torch=FAIL; vs master=FAIL; band=PASS (1.498x) |
| Mini | 4096 | strict self=FAIL | strict self=FAIL; vs Torch=FAIL; band=FAIL (4.252x) | strict self=FAIL; vs Torch=FAIL; vs master=FAIL; band=FAIL (4.346x) |
| Pro | 96 | strict self=PASS | strict self=PASS; vs Torch=PASS; band=PASS (1.678x) | strict self=PASS; vs Torch=PASS; vs master=PASS; band=PASS (1.709x) |
| Pro | 512 | strict self=N/A | strict self=PASS; vs Torch=N/A; band=N/A | strict self=PASS; vs Torch=N/A; vs master=PASS; band=N/A |
| Pro | 4096 | strict self=N/A | strict self=N/A; vs Torch=N/A; band=N/A | strict self=N/A; vs Torch=N/A; vs master=N/A; band=N/A |

## Three accuracy columns

Each field: worst pairwise max-absolute difference over same-frame outputs; self includes within/across-process repeats. Scientific band multiplier = worst deviation from compiled TORCH / TORCH own worst repeat max_abs, separately per field/model/size/system. <=3 is the meeting criterion, not a replacement for the fixed strict elementwise gate. Zero repeat band requires zero cross difference. Energy/virial eV, force eV/Angstrom, stress eV/Angstrom^3.

| Model/system | Atoms | Output | Compiled Torch | Current master | Candidate |
| --- | ---: | --- | --- | --- | --- |
| Neo | 96 | energy | self max_abs=1.222e-05 | self max_abs=1.073e-05; vs Torch=1.115e-05; band×0.912 | self max_abs=6.855e-06; vs Torch=1.127e-05; band×0.922 |
| Neo | 96 | forces | self max_abs=4.93e-06 | self max_abs=4.474e-06; vs Torch=7.196e-06; band×1.46 | self max_abs=4.369e-06; vs Torch=7.185e-06; band×1.46 |
| Neo | 96 | virial | self max_abs=2.222e-05 | self max_abs=2.443e-05; vs Torch=4.824e-05; band×2.17 | self max_abs=2.69e-05; vs Torch=5.232e-05; band×2.35 |
| Neo | 96 | stress | self max_abs=4.08e-08 | self max_abs=4.486e-08; vs Torch=8.858e-08; band×2.17 | self max_abs=4.938e-08; vs Torch=9.607e-08; band×2.35 |
| Neo | 512 | energy | self max_abs=1.776e-05 | self max_abs=2.247e-05; vs Torch=2.021e-05; band×1.14 | self max_abs=2.056e-05; vs Torch=2.718e-05; band×1.53 |
| Neo | 512 | forces | self max_abs=4.901e-06 | self max_abs=5.144e-06; vs Torch=7.15e-06; band×1.46 | self max_abs=5.012e-06; vs Torch=7.02e-06; band×1.43 |
| Neo | 512 | virial | self max_abs=6.951e-05 | self max_abs=6.508e-05; vs Torch=0.0001673; band×2.41 | self max_abs=5.708e-05; vs Torch=0.0001572; band×2.26 |
| Neo | 512 | stress | self max_abs=2.393e-08 | self max_abs=2.24e-08; vs Torch=5.758e-08; band×2.41 | self max_abs=1.965e-08; vs Torch=5.411e-08; band×2.26 |
| Neo | 4096 | energy | OOM | self max_abs=5.96e-05; Torch ref unavailable | self max_abs=5.788e-05; Torch ref unavailable |
| Neo | 4096 | forces | OOM | self max_abs=5.564e-06; Torch ref unavailable | self max_abs=5.939e-06; Torch ref unavailable |
| Neo | 4096 | virial | OOM | self max_abs=0.0001374; Torch ref unavailable | self max_abs=0.0002537; Torch ref unavailable |
| Neo | 4096 | stress | OOM | self max_abs=5.913e-09; Torch ref unavailable | self max_abs=1.092e-08; Torch ref unavailable |
| NeoCu | 96 | energy | self max_abs=3.636e-06 | self max_abs=4.59e-06; vs Torch=6.855e-06; band×1.89 | self max_abs=4.411e-06; vs Torch=5.782e-06; band×1.59 |
| NeoCu | 96 | forces | self max_abs=6.306e-07 | self max_abs=6.496e-07; vs Torch=6.391e-07; band×1.01 | self max_abs=6.476e-07; vs Torch=6.297e-07; band×0.999 |
| NeoCu | 96 | virial | self max_abs=1.361e-05 | self max_abs=1.663e-05; vs Torch=2.707e-05; band×1.99 | self max_abs=1.739e-05; vs Torch=2.837e-05; band×2.08 |
| NeoCu | 96 | stress | self max_abs=1.2e-08 | self max_abs=1.466e-08; vs Torch=2.386e-08; band×1.99 | self max_abs=1.533e-08; vs Torch=2.5e-08; band×2.08 |
| NeoCu | 512 | energy | self max_abs=1.138e-05 | self max_abs=1.007e-05; vs Torch=1.919e-05; band×1.69 | self max_abs=1.085e-05; vs Torch=1.86e-05; band×1.63 |
| NeoCu | 512 | forces | self max_abs=6.573e-07 | self max_abs=6.11e-07; vs Torch=7.357e-07; band×1.12 | self max_abs=7.186e-07; vs Torch=7.158e-07; band×1.09 |
| NeoCu | 512 | virial | self max_abs=4.521e-05 | self max_abs=3.137e-05; vs Torch=0.000113; band×2.5 | self max_abs=4.702e-05; vs Torch=0.0001295; band×2.86 |
| NeoCu | 512 | stress | self max_abs=7.471e-09 | self max_abs=5.184e-09; vs Torch=1.867e-08; band×2.5 | self max_abs=7.77e-09; vs Torch=2.139e-08; band×2.86 |
| NeoCu | 4096 | energy | self max_abs=2.599e-05 | self max_abs=4.148e-05; vs Torch=8.726e-05; band×3.36 | self max_abs=3.821e-05; vs Torch=8.494e-05; band×3.27 |
| NeoCu | 4096 | forces | self max_abs=7.152e-07 | self max_abs=7.069e-07; vs Torch=8.014e-07; band×1.12 | self max_abs=7.169e-07; vs Torch=7.932e-07; band×1.11 |
| NeoCu | 4096 | virial | self max_abs=0.0001036 | self max_abs=0.0001054; vs Torch=0.0006932; band×6.69 | self max_abs=0.0001111; vs Torch=0.000695; band×6.71 |
| NeoCu | 4096 | stress | self max_abs=2.14e-09 | self max_abs=2.178e-09; vs Torch=1.432e-08; band×6.69 | self max_abs=2.295e-09; vs Torch=1.435e-08; band×6.71 |
| Nano | 96 | energy | self max_abs=2.658e-05 | self max_abs=1.878e-05; vs Torch=3.499e-05; band×1.32 | self max_abs=2.122e-05; vs Torch=3.982e-05; band×1.5 |
| Nano | 96 | forces | self max_abs=1.136e-05 | self max_abs=1.22e-05; vs Torch=1.286e-05; band×1.13 | self max_abs=1.15e-05; vs Torch=1.269e-05; band×1.12 |
| Nano | 96 | virial | self max_abs=7.839e-05 | self max_abs=8.811e-05; vs Torch=9.567e-05; band×1.22 | self max_abs=8.601e-05; vs Torch=9.163e-05; band×1.17 |
| Nano | 96 | stress | self max_abs=1.439e-07 | self max_abs=1.618e-07; vs Torch=1.757e-07; band×1.22 | self max_abs=1.579e-07; vs Torch=1.682e-07; band×1.17 |
| Nano | 512 | energy | self max_abs=6.02e-05 | self max_abs=6.39e-05; vs Torch=0.0001107; band×1.84 | self max_abs=5.358e-05; vs Torch=0.0001102; band×1.83 |
| Nano | 512 | forces | self max_abs=1.488e-05 | self max_abs=1.347e-05; vs Torch=1.477e-05; band×0.992 | self max_abs=1.348e-05; vs Torch=1.325e-05; band×0.89 |
| Nano | 512 | virial | self max_abs=0.000201 | self max_abs=0.0001863; vs Torch=0.0002575; band×1.28 | self max_abs=0.0002203; vs Torch=0.000256; band×1.27 |
| Nano | 512 | stress | self max_abs=6.921e-08 | self max_abs=6.415e-08; vs Torch=8.866e-08; band×1.28 | self max_abs=7.585e-08; vs Torch=8.813e-08; band×1.27 |
| Nano | 4096 | energy | self max_abs=0.0001612 | self max_abs=0.0001924; vs Torch=0.0004887; band×3.03 | self max_abs=0.0002133; vs Torch=0.0005112; band×3.17 |
| Nano | 4096 | forces | self max_abs=1.484e-05 | self max_abs=1.563e-05; vs Torch=1.613e-05; band×1.09 | self max_abs=1.605e-05; vs Torch=1.554e-05; band×1.05 |
| Nano | 4096 | virial | self max_abs=0.0004504 | self max_abs=0.0005898; vs Torch=0.0007157; band×1.59 | self max_abs=0.0005332; vs Torch=0.0007943; band×1.76 |
| Nano | 4096 | stress | self max_abs=1.938e-08 | self max_abs=2.538e-08; vs Torch=3.08e-08; band×1.59 | self max_abs=2.294e-08; vs Torch=3.418e-08; band×1.76 |
| Mini | 96 | energy | self max_abs=1.246e-05 | self max_abs=1.049e-05; vs Torch=1.836e-05; band×1.47 | self max_abs=1.74e-05; vs Torch=1.651e-05; band×1.33 |
| Mini | 96 | forces | self max_abs=8.618e-06 | self max_abs=7.551e-06; vs Torch=1.08e-05; band×1.25 | self max_abs=8.28e-06; vs Torch=1.135e-05; band×1.32 |
| Mini | 96 | virial | self max_abs=6.538e-05 | self max_abs=4.644e-05; vs Torch=8.237e-05; band×1.26 | self max_abs=7.174e-05; vs Torch=9.37e-05; band×1.43 |
| Mini | 96 | stress | self max_abs=1.2e-07 | self max_abs=8.526e-08; vs Torch=1.512e-07; band×1.26 | self max_abs=1.317e-07; vs Torch=1.72e-07; band×1.43 |
| Mini | 512 | energy | self max_abs=3.606e-05 | self max_abs=3.594e-05; vs Torch=3.952e-05; band×1.1 | self max_abs=2.813e-05; vs Torch=3.04e-05; band×0.843 |
| Mini | 512 | forces | self max_abs=8.601e-06 | self max_abs=8.582e-06; vs Torch=1.21e-05; band×1.41 | self max_abs=9.532e-06; vs Torch=1.289e-05; band×1.5 |
| Mini | 512 | virial | self max_abs=0.0001708 | self max_abs=0.0001244; vs Torch=0.0002595; band×1.52 | self max_abs=0.0001445; vs Torch=0.0002485; band×1.45 |
| Mini | 512 | stress | self max_abs=5.882e-08 | self max_abs=4.283e-08; vs Torch=8.933e-08; band×1.52 | self max_abs=4.975e-08; vs Torch=8.555e-08; band×1.45 |
| Mini | 4096 | energy | self max_abs=7.707e-05 | self max_abs=0.0001156; vs Torch=0.0001388; band×1.8 | self max_abs=8.416e-05; vs Torch=0.0001465; band×1.9 |
| Mini | 4096 | forces | self max_abs=1.004e-05 | self max_abs=9.328e-06; vs Torch=1.306e-05; band×1.3 | self max_abs=1.007e-05; vs Torch=1.398e-05; band×1.39 |
| Mini | 4096 | virial | self max_abs=0.0003499 | self max_abs=0.0003119; vs Torch=0.001488; band×4.25 | self max_abs=0.0003646; vs Torch=0.001521; band×4.35 |
| Mini | 4096 | stress | self max_abs=1.506e-08 | self max_abs=1.342e-08; vs Torch=6.403e-08; band×4.25 | self max_abs=1.569e-08; vs Torch=6.544e-08; band×4.35 |
| Pro | 96 | energy | self max_abs=4.083e-06 | self max_abs=3.368e-06; vs Torch=5.037e-06; band×1.23 | self max_abs=3.338e-06; vs Torch=5.871e-06; band×1.44 |
| Pro | 96 | forces | self max_abs=1.667e-06 | self max_abs=1.153e-06; vs Torch=1.919e-06; band×1.15 | self max_abs=1.144e-06; vs Torch=1.691e-06; band×1.01 |
| Pro | 96 | virial | self max_abs=9.613e-06 | self max_abs=8.763e-06; vs Torch=1.613e-05; band×1.68 | self max_abs=7.585e-06; vs Torch=1.643e-05; band×1.71 |
| Pro | 96 | stress | self max_abs=1.765e-08 | self max_abs=1.609e-08; vs Torch=2.962e-08; band×1.68 | self max_abs=1.393e-08; vs Torch=3.017e-08; band×1.71 |
| Pro | 512 | energy | OOM | self max_abs=1.019e-05; Torch ref unavailable | self max_abs=8.345e-06; Torch ref unavailable |
| Pro | 512 | forces | OOM | self max_abs=1.36e-06; Torch ref unavailable | self max_abs=1.497e-06; Torch ref unavailable |
| Pro | 512 | virial | OOM | self max_abs=2.018e-05; Torch ref unavailable | self max_abs=1.641e-05; Torch ref unavailable |
| Pro | 512 | stress | OOM | self max_abs=6.947e-09; Torch ref unavailable | self max_abs=5.648e-09; Torch ref unavailable |
| Pro | 4096 | energy | OOM | OOM | OOM |
| Pro | 4096 | forces | OOM | OOM | OOM |
| Pro | 4096 | virial | OOM | OOM | OOM |
| Pro | 4096 | stress | OOM | OOM | OOM |

Strict per-element atol1e-5/rtol1e-4 and finite results, mean_abs/max_relative and within/across detail remain in ACCURACY_TABLE.md and summary.json. Pure-reference OOM prevents band acceptance. Band PASS does not silently erase strict FAIL.

## Timing and memory — process medians and worst capacity

Compilation excluded from timing, included in first/full memory. This independent-process protocol does not freeze all generated autotune choices: no stable isolated speedup is inferred solely from these percentages. GB=1e9B. Both allocated/reserved raw bytes are retained in result.json. Numbers are preliminary until 3 processes per arm complete.

| Model/system | Atoms | Mode | Processes | Wall P50 ms | Wall P90 ms | Event P50 ms | First alloc GB | Full peak incl first GB | Steady alloc GB |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Neo | 96 | TORCH | 3/3 | 13.075581 | 13.151851 | 13.030096 | 1.118502 | 1.118502 | 1.017839 |
| Neo | 96 | BASE | 3/3 | 17.992521 | 18.385345 | 17.950175 | 0.646372 | 0.646372 | 0.545709 |
| Neo | 96 | ECS | 3/3 | 18.035012 | 18.224658 | 17.991696 | 0.658145 | 0.658145 | 0.557481 |
| Neo | 512 | TORCH | 3/3 | 46.979462 | 47.018433 | 46.936768 | 5.331018 | 5.331018 | 5.331018 |
| Neo | 512 | BASE | 3/3 | 23.472956 | 23.531686 | 23.427423 | 2.809001 | 2.809001 | 2.809001 |
| Neo | 512 | ECS | 3/3 | 23.494863 | 23.648258 | 23.449760 | 2.871368 | 2.871368 | 2.871368 |
| Neo | 4096 | BASE | 3/3 | 156.512587 | 156.684338 | 156.465652 | 22.313645 | 22.313645 | 22.313645 |
| Neo | 4096 | ECS | 3/3 | 156.594862 | 156.777619 | 156.533646 | 22.806378 | 22.806378 | 22.806378 |
| NeoCu | 96 | TORCH | 3/3 | 12.232151 | 12.297081 | 12.188800 | 0.541385 | 0.541385 | 0.440721 |
| NeoCu | 96 | BASE | 3/3 | 18.154909 | 18.366497 | 18.111904 | 0.353757 | 0.353757 | 0.253004 |
| NeoCu | 96 | ECS | 3/3 | 18.259421 | 18.342752 | 18.215392 | 0.355033 | 0.355033 | 0.254369 |
| NeoCu | 512 | TORCH | 3/3 | 22.897386 | 22.935968 | 22.853473 | 2.253040 | 2.253040 | 2.253040 |
| NeoCu | 512 | BASE | 3/3 | 18.317986 | 18.392029 | 18.274000 | 1.249555 | 1.249555 | 1.249555 |
| NeoCu | 512 | ECS | 3/3 | 18.244627 | 18.391290 | 18.201713 | 1.254748 | 1.254748 | 1.254748 |
| NeoCu | 4096 | TORCH | 3/3 | 160.911316 | 161.083811 | 160.864113 | 17.866908 | 17.866908 | 17.866908 |
| NeoCu | 4096 | BASE | 3/3 | 80.131557 | 80.263306 | 80.085968 | 9.842755 | 9.842755 | 9.842755 |
| NeoCu | 4096 | ECS | 3/3 | 80.314757 | 80.400489 | 80.269539 | 9.873422 | 9.873422 | 9.873422 |
| Nano | 96 | TORCH | 3/3 | 10.842765 | 10.894773 | 10.799536 | 0.267505 | 0.267505 | 0.168914 |
| Nano | 96 | BASE | 3/3 | 10.827782 | 10.899461 | 10.785792 | 0.204831 | 0.204831 | 0.104168 |
| Nano | 96 | ECS | 3/3 | 10.780998 | 10.826602 | 10.738672 | 0.204831 | 0.204831 | 0.104168 |
| Nano | 512 | TORCH | 3/3 | 11.856695 | 11.910365 | 11.813136 | 0.845872 | 0.845872 | 0.845872 |
| Nano | 512 | BASE | 3/3 | 11.192140 | 11.325888 | 11.147888 | 0.500028 | 0.500028 | 0.500028 |
| Nano | 512 | ECS | 3/3 | 11.160740 | 11.249917 | 11.116624 | 0.500028 | 0.500028 | 0.500028 |
| Nano | 4096 | TORCH | 3/3 | 56.467913 | 56.592532 | 56.422001 | 6.678115 | 6.678115 | 6.678115 |
| Nano | 4096 | BASE | 3/3 | 33.593906 | 33.720488 | 33.547775 | 3.910505 | 3.910505 | 3.910505 |
| Nano | 4096 | ECS | 3/3 | 33.589447 | 33.682449 | 33.544529 | 3.910505 | 3.910505 | 3.910505 |
| Mini | 96 | TORCH | 3/3 | 11.962016 | 12.010488 | 11.917488 | 0.400473 | 0.400473 | 0.299810 |
| Mini | 96 | BASE | 3/3 | 11.099674 | 11.319123 | 11.057840 | 0.286303 | 0.286303 | 0.190500 |
| Mini | 96 | ECS | 3/3 | 11.180259 | 11.242529 | 11.135952 | 0.286303 | 0.286303 | 0.190500 |
| Mini | 512 | TORCH | 3/3 | 18.303323 | 18.349090 | 18.260481 | 1.528662 | 1.528662 | 1.528662 |
| Mini | 512 | BASE | 3/3 | 12.184330 | 12.276598 | 12.140736 | 0.944773 | 0.945822 | 0.945822 |
| Mini | 512 | ECS | 3/3 | 12.208581 | 12.249058 | 12.166288 | 0.944773 | 0.945822 | 0.945822 |
| Mini | 4096 | TORCH | 3/3 | 122.270842 | 122.379773 | 122.217281 | 12.115697 | 12.115697 | 12.115697 |
| Mini | 4096 | BASE | 3/3 | 61.097768 | 61.183442 | 61.052145 | 7.452142 | 7.453190 | 7.453190 |
| Mini | 4096 | ECS | 3/3 | 60.989576 | 61.053372 | 60.944431 | 7.452142 | 7.453190 | 7.453190 |
| Pro | 96 | TORCH | 3/3 | 125.952058 | 126.149409 | 125.908112 | 11.443602 | 11.443602 | 11.443602 |
| Pro | 96 | BASE | 3/3 | 66.992101 | 67.231234 | 66.948528 | 5.139665 | 5.139665 | 5.039002 |
| Pro | 96 | ECS | 3/3 | 67.021606 | 67.312684 | 66.977631 | 5.139665 | 5.139665 | 5.039002 |
| Pro | 512 | BASE | 3/3 | 330.464768 | 331.377085 | 330.418015 | 25.395441 | 25.395441 | 25.395441 |
| Pro | 512 | ECS | 3/3 | 330.612567 | 331.352630 | 330.561584 | 25.395441 | 25.395441 | 25.395441 |

## Route and acceptance limits

Nano/Mini/Pro candidate is native fallback, not demonstrated ECS acceleration. Neo dispatch must be checked against generated wrappers; current single-cohort specialization must not be described as 80 runtime batches. Pro-MPtrj is not OMat24-Pro; original checkpoint unmodified, reader-only legacy metadata alias applies equally to all arms. H20/PT2/C++/training and unmeasured larger models are not accepted by this matrix.

Upstream review also requires focused guard/alias tests, no new public fine-grained development switch for a merge-ready path, freeze-time policy/selection where applicable, optional-Triton CPU safety and real CI approval. Current port keeps the old default-off experimental switch pending policy resolution. Do not submit as ready-to-merge.

## Candidate vs current-master changes

Positive percentage means reduction/improvement; negative means increase. Medians of independent process quantiles, not a controlled paired significance estimate. Fallback timing variation is not attributed to ECS.

| Model/system | Atoms | Time reduction % | First-inclusive full alloc reduction % | Steady alloc reduction % | Candidate route |
| --- | ---: | ---: | ---: | ---: | --- |
| Neo | 96 | -0.2362 | -1.8213 | -2.1573 | single-cohort specialized, not bounded runtime batching |
| Neo | 512 | -0.0933 | -2.2202 | -2.2202 | single-cohort specialized, not bounded runtime batching |
| Neo | 4096 | -0.0526 | -2.2082 | -2.2082 | single-cohort specialized, not bounded runtime batching |
| NeoCu | 96 | -0.5757 | -0.3605 | -0.5395 | single-cohort specialized, not bounded runtime batching |
| NeoCu | 512 | 0.4005 | -0.4156 | -0.4156 | single-cohort specialized, not bounded runtime batching |
| NeoCu | 4096 | -0.2286 | -0.3116 | -0.3116 | single-cohort specialized, not bounded runtime batching |
| Nano | 96 | 0.4321 | 0.0000 | 0.0000 | native fallback |
| Nano | 512 | 0.2806 | 0.0000 | 0.0000 | native fallback |
| Nano | 4096 | 0.0133 | 0.0000 | 0.0000 | native fallback |
| Mini | 96 | -0.7260 | 0.0000 | 0.0000 | native fallback |
| Mini | 512 | -0.1990 | 0.0000 | 0.0000 | native fallback |
| Mini | 4096 | 0.1771 | 0.0000 | 0.0000 | native fallback |
| Pro | 96 | -0.0440 | 0.0000 | 0.0000 | native fallback |
| Pro | 512 | -0.0447 | 0.0000 | 0.0000 | native fallback |
| Pro | 4096 | N/A | N/A | N/A | incomplete/OOM |
