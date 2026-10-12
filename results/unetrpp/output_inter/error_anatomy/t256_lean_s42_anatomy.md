# Error anatomy — t256_lean_s42 (tau 0.75, recall 0.578)

| FN stratum | px | % of FN | % of GT |
|---|---|---|---|
| dead_label_components | 22,169 | 2.2% | 0.9% |
| boundary_band_<= 4px | 135,750 | 13.5% | 5.7% |
| low_contrast_<3.0 | 128,060 | 12.7% | 5.4% |
| UNEXPLAINED (attainable headroom) | 720,243 | 71.6% | 30.2% |

FP: 150,890 px total — 36,023 within 4px of GT (boundary), 114,867 in 342 isolated islands.
