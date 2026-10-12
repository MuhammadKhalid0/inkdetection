# Error anatomy — n_cssa_t192_r1_s42 (tau 0.75, recall 0.552)

| FN stratum | px | % of FN | % of GT |
|---|---|---|---|
| dead_label_components | 9,117 | 0.9% | 0.4% |
| boundary_band_<= 4px | 135,543 | 12.7% | 5.7% |
| low_contrast_<3.0 | 129,826 | 12.2% | 5.4% |
| UNEXPLAINED (attainable headroom) | 792,370 | 74.3% | 33.2% |

FP: 240,764 px total — 39,814 within 4px of GT (boundary), 200,950 in 409 isolated islands.
