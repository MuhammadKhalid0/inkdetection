# Error anatomy — n_srpb_t192_r1_s42 (tau 0.7, recall 0.540)

| FN stratum | px | % of FN | % of GT |
|---|---|---|---|
| dead_label_components | 22,169 | 2.0% | 0.9% |
| boundary_band_<= 4px | 134,225 | 12.2% | 5.6% |
| low_contrast_<3.0 | 115,531 | 10.5% | 4.8% |
| UNEXPLAINED (attainable headroom) | 824,075 | 75.2% | 34.6% |

FP: 197,869 px total — 38,898 within 4px of GT (boundary), 158,971 in 331 isolated islands.
