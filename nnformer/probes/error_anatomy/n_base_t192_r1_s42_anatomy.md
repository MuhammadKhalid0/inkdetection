# Error anatomy — n_base_t192_r1_s42 (tau 0.85, recall 0.537)

| FN stratum | px | % of FN | % of GT |
|---|---|---|---|
| dead_label_components | 22,169 | 2.0% | 0.9% |
| boundary_band_<= 4px | 139,682 | 12.7% | 5.9% |
| low_contrast_<3.0 | 122,314 | 11.1% | 5.1% |
| UNEXPLAINED (attainable headroom) | 818,851 | 74.2% | 34.4% |

FP: 227,781 px total — 33,905 within 4px of GT (boundary), 193,876 in 511 isolated islands.
