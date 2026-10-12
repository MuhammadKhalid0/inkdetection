# Error anatomy — nnf_s_d64_t256_s42 (tau 0.9, recall 0.527)

| FN stratum | px | % of FN | % of GT |
|---|---|---|---|
| dead_label_components | 9,117 | 0.8% | 0.4% |
| boundary_band_<= 4px | 146,435 | 13.0% | 6.1% |
| low_contrast_<3.0 | 139,438 | 12.4% | 5.8% |
| UNEXPLAINED (attainable headroom) | 831,931 | 73.8% | 34.9% |

FP: 168,622 px total — 30,178 within 4px of GT (boundary), 138,444 in 373 isolated islands.
