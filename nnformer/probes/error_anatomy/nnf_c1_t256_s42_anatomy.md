# Error anatomy — nnf_c1_t256_s42 (tau 0.85, recall 0.588)

| FN stratum | px | % of FN | % of GT |
|---|---|---|---|
| dead_label_components | 9,117 | 0.9% | 0.4% |
| boundary_band_<= 4px | 134,225 | 13.7% | 5.6% |
| low_contrast_<3.0 | 130,172 | 13.3% | 5.5% |
| UNEXPLAINED (attainable headroom) | 707,415 | 72.1% | 29.7% |

FP: 230,643 px total — 40,210 within 4px of GT (boundary), 190,433 in 514 isolated islands.
