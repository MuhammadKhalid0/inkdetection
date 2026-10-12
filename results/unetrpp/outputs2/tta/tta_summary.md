# TTA re-scoring (8-fold dihedral, existing checkpoints)

| run | raw F0.5 TTA | Delta raw | post F0.5 TTA | Delta post | IoU_ink post |
|---|---|---|---|---|---|
| lean_t192_s42 | 0.7835 | +0.0051 | 0.8000 | +0.0043 | 0.5407 |
| lean_t192_s43 | 0.7762 | +0.0025 | 0.7925 | +0.0056 | 0.4980 |
| lean_t192_s44 | 0.7712 | -0.0021 | 0.7974 | +0.0005 | 0.5235 |
| a2bs_t192 | 0.7805 | +0.0070 | 0.7905 | +0.0049 | 0.5609 |
| a2bs_t192_s43 | 0.7830 | -0.0001 | 0.8009 | +0.0046 | 0.5356 |
| a2bs_t192_s44 | 0.7833 | +0.0014 | 0.7988 | -0.0028 | 0.5472 |

Mean Delta(post) over lean seeds: +0.0035 -> ADOPT (gate +0.003)
