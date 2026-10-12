# nnFormer — master results (auto-generated)

All numbers: masked swept-F0.5 on val 20241113090990, tau selected on val,
post = 8-connected min-size filter. Noise floor: single-seed |d| <= 0.008.

## All runs

| run | labels | raw | tau | post | P | R | best ep | overfit? |
|---|---|---|---|---|---|---|---|---|
| n0_sanity_t64 | ink_reg.npy | 0.2000 | 0.35 | 0.0000 | 0.000 | 0.000 | 1 | no |
| n_abs_t192_s42 | ink_reg2.npy | 0.7682 | 0.85 | 0.7847 | 0.848 | 0.605 | 48 | no |
| n_base_t128_r1_s42 | ink_reg.npy | 0.7518 | 0.80 | 0.7627 | 0.850 | 0.540 | 54 | no |
| n_base_t192_r1_s42 | ink_reg.npy | 0.7581 | 0.85 | 0.7747 | 0.877 | 0.529 | 59 | no |
| n_base_t192_r1_s43 | ink_reg.npy | 0.7687 | 0.85 | 0.7919 | 0.871 | 0.580 | 54 | no |
| n_base_t192_r1_s44 | ink_reg.npy | 0.7703 | 0.80 | 0.7844 | 0.870 | 0.563 | 55 | no |
| n_base_t192_s42 | ink_reg2.npy | 0.7780 | 0.80 | 0.7880 | 0.879 | 0.557 | 52 | no |
| n_cssa_t128_r1_s42 | ink_reg.npy | 0.7539 | 0.85 | 0.7642 | 0.818 | 0.605 | 56 | no |
| n_cssa_t192_r1_s42 | ink_reg.npy | 0.7618 | 0.75 | 0.7766 | 0.838 | 0.601 | 42 | no |
| n_embedz2_t192_s42 | ink_reg2.npy | 0.7665 | 0.80 | 0.7951 | 0.857 | 0.617 | 51 | no |
| n_lr5e5_t192_s42 | ink_reg2.npy | 0.7785 | 0.80 | 0.7937 | 0.864 | 0.600 | 45 | no |
| n_lvonly_t192_s42 | ink_reg2.npy | 0.7833 | 0.75 | 0.7910 | 0.875 | 0.571 | 36 | YES |
| n_mean_t192_s42 | ink_reg2.npy | 0.7739 | 0.80 | 0.7953 | 0.859 | 0.613 | 51 | no |
| n_skipconv_t192_s42 | ink_reg2.npy | 0.7778 | 0.80 | 0.7850 | 0.882 | 0.545 | 54 | no |
| n_srpb_aux_t192_r1_s42 | ink_reg.npy | 0.7607 | 0.75 | 0.7747 | 0.885 | 0.517 | 50 | no |
| n_srpb_t192_r1_s42 | ink_reg.npy | 0.7708 | 0.70 | 0.7796 | 0.894 | 0.516 | 50 | no |
| n_srpb_t192_r1_s43 | ink_reg.npy | 0.7724 | 0.80 | 0.7852 | 0.876 | 0.555 | 56 | no |
| n_srpb_t192_r1_s44 | ink_reg.npy | 0.7733 | 0.85 | 0.7844 | 0.837 | 0.627 | 49 | no |
| n_vffc_t192_r1_s42 | ink_reg.npy | 0.7844 | 0.80 | 0.7992 | 0.872 | 0.599 | 48 | no |
| nnf_c1_t256_s42 | ink_reg.npy | 0.7843 | 0.85 | 0.8122 | 0.862 | 0.660 | 59 | no |
| nnf_c1_t256_s43 | ink_reg.npy | 0.7776 | 0.85 | 0.7932 | 0.879 | 0.570 | 52 | no |
| nnf_c1_t256_s44 | ink_reg.npy | 0.7785 | 0.80 | 0.7960 | 0.877 | 0.581 | 54 | no |
| nnf_f_d64_t256_s42 | ink_reg.npy | 0.7652 | 0.85 | 0.7807 | 0.895 | 0.517 | 53 | no |
| nnf_f_t256_s42 | ink_reg.npy | 0.7671 | 0.80 | 0.7806 | 0.854 | 0.581 | 37 | YES |
| nnf_m_t256_s42 | ink_reg.npy | 0.7613 | 0.80 | 0.7753 | 0.867 | 0.545 | 44 | no |
| nnf_s_d64_t256_e90_s42 | ink_reg.npy | 0.7809 | 0.80 | 0.8102 | 0.882 | 0.610 | 54 | YES |
| nnf_s_d64_t256_s42 | ink_reg.npy | 0.7747 | 0.90 | 0.7884 | 0.859 | 0.593 | 50 | no |
| nnf_s_d64_t256_s43 | ink_reg.npy | 0.7770 | 0.80 | 0.7940 | 0.885 | 0.563 | 57 | no |
| nnf_s_d64_t256_s44 | ink_reg.npy | 0.7960 | 0.85 | 0.8120 | 0.887 | 0.606 | 50 | no |
| nnf_s_t256_s42 | ink_reg.npy | 0.7677 | 0.80 | 0.7868 | 0.883 | 0.549 | 59 | no |

## Gate: CSSA (contribution C2) — pre-registered 2x2

| | t128 | t192 | d(128->192) |
|---|---|---|---|
| stock | 0.7518 | 0.7581 | +0.0063 |
| CSSA | 0.7539 | 0.7618 | +0.0080 |
| d(CSSA-stock) | +0.0020 | +0.0037 | |

Stock t128->t192 gap = **+0.0063**; CSSA@t128 closes **+0.0020** = **32%** of it.

**VERDICT: FAIL (informative null)** — the pre-written reading applies: letterform context must enter at input resolution; it cannot be retrieved from coarse features after the fact. This *strengthens* the tile-ladder finding. Report the gamma trajectory as mechanism evidence; do NOT spend promotion seeds.

## Gate: SR-PB (contribution C1) — band vs band

| | raw | post |
|---|---|---|
| stock t192 (n=3) | 0.7657 ± 0.0066 | 0.7836 ± 0.0086 |
| SR-PB t192 (n=3) | 0.7722 ± 0.0013 | 0.7831 ± 0.0030 |
| **delta (band vs band)** | **+0.0065** | **-0.0006** |

Stock seeds individually: s42 0.7581, s43 0.7687, s44 0.7703.
**VERDICT: NEUTRAL on F0.5** — both deltas inside the ±0.008 floor. What survives: seed-variance reduction (raw sd 0.0013 vs 0.0066) and the Inkalyzer depth-broadening mechanism. The depth-offset stress curve still uses ONLY base s42 (the weakest seed) as baseline — replicate on s43/s44 before claiming.

Stress runs present: n_base_t192_r1_s42, n_srpb_t192_r1_s42, n_srpb_t192_r1_s43, n_cssa_t192_r1_s42

## Benchmark row (cross-model headline)

| model | raw | post |
|---|---|---|
| UNETR++ c1_t256 (3 seeds) | 0.8111 | 0.8190 |
| UNETR++ t256_lean_s42 | 0.8085 | 0.8181 |
| UNETR++ lean_t192_s42 | 0.7784 | 0.7957 |
| UNETR++ r4 pipeline | 0.7909 | 0.7986 |
| **nnFormer nnf_s_d64_t256 (3 seed(s))** | 0.7825 ± 0.0117 | 0.7981 ± 0.0123 |

Gap to UNETR++ c1_t256 (0.8190 post): **-0.0070** at best nnFormer seed.

## Budget test (N2 flag: val still rising at e59)

e90: raw 0.7809 (best ep 54) vs e60 same seed: raw 0.7747 (best ep 50) -> **+0.0062**

UNETR++ precedent: e90 < e60 (0.8070 vs 0.8141) — the 60-epoch cosine was confirmed there. This tests whether nnFormer differs.
