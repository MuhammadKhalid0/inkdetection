# Error anatomy — lean_t192_s42 (tau 0.75, recall 0.537)

| FN stratum | px | % of FN | % of GT |
|---|---|---|---|
| dead_label_components | 22,169 | 2.0% | 0.9% |
| boundary_band_<= 4px | 136,874 | 12.4% | 5.7% |
| low_contrast_<3.0 | 146,520 | 13.3% | 6.1% |
| UNEXPLAINED (attainable headroom) | 797,027 | 72.3% | 33.4% |

FP: 173,853 px total — 35,962 within 4px of GT (boundary), 137,891 in 302 isolated islands.
