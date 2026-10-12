# Runs that hit the 24 h A40 wall (archived 2026-09-12, nothing deleted)

| run | epochs reached | best epoch | status |
|---|---|---|---|
| sf_base_t320_s42 | 45/60 | 33 | re-run on A100 as a resumable chain |
| sf_base_d96_t320_s42 | 45/60 | 29 | re-run on A100 as a resumable chain |
| sf_m_t256_s42 | 38/60 | 32 | re-run on A100 as a resumable chain |
| sf_m_t320_s42 | 22/60 | 21 | re-run on A100 as a resumable chain |
| sf_l_t256_s42 | 10/60 | 8 | DEFERRED under the plan's scope guard (~6 GPU-days even with chaining) |

These folders have best.pt / summary.json / val_prob_best.npy / train_log.csv but no
postprocess.json. They pre-date resume support, so they have no last.pt and cannot be
continued; the A100 re-runs start from epoch 0. Kept for the record and as a fallback.
Measured A40 time per epoch: t320 SF-S 32 min, SF-M t256 38 min, SF-M t320 65 min, SF-L t256 144 min.
