# SegFormer3D — master results (auto-generated)

Masked swept-F0.5 on val 20241113090990 vs `ink_reg.npy`; tau selected on val;
post = 8-connected min-size filter. Noise floor: single-seed |d| <= 0.008.

## All runs

| run | raw | tau | post | P | R | best ep | overfit? |
|---|---|---|---|---|---|---|---|
| s0_sanity_t64 | 0.2598 | 0.35 | 0.0000 | 0.000 | 0.000 | 1 | no |
| sf_25d_t256_s42 | 0.7254 | 0.85 | 0.7379 | 0.866 | 0.463 | 50 | no |
| sf_base_d96_t320_s42 | 0.8006 | 0.75 | 0.8106 | 0.876 | 0.623 | 38 | YES |
| sf_base_t128_s42 | 0.7578 | 0.85 | 0.7787 | 0.819 | 0.652 | 49 | no |
| sf_base_t192_s42 | 0.7857 | 0.85 | 0.8092 | 0.893 | 0.588 | 48 | no |
| sf_base_t256_s42 | 0.7948 | 0.85 | 0.8204 | 0.886 | 0.632 | 39 | YES |
| sf_base_t320_s42 | 0.8003 | 0.75 | 0.8191 | 0.899 | 0.605 | 38 | no |
| sf_base_t64_s42 | 0.6816 | 0.80 | 0.6932 | 0.782 | 0.476 | 49 | no |
| sf_dcm_abs_t192_s42 | 0.7869 | 0.80 | 0.8098 | 0.877 | 0.621 | 32 | YES |
| sf_dcm_surf_t192_s42 | 0.7851 | 0.70 | 0.8028 | 0.875 | 0.604 | 26 | YES |
| sf_dec128_t192_s42 | 0.7810 | 0.85 | 0.8036 | 0.888 | 0.582 | 34 | no |
| sf_deep_t192_s42 | 0.7777 | 0.85 | 0.7986 | 0.891 | 0.565 | 18 | no |
| sf_depth65_t192_s42 | 0.7874 | 0.75 | 0.8028 | 0.894 | 0.570 | 30 | YES |
| sf_dsm_t192_s42 | 0.7862 | 0.65 | 0.8142 | 0.890 | 0.608 | 32 | no |
| sf_finlow_t192_s42 | 0.7915 | 0.80 | 0.8089 | 0.876 | 0.620 | 37 | YES |
| sf_lr5e5_t192_s42 | 0.7869 | 0.80 | 0.8092 | 0.911 | 0.559 | 43 | no |
| sf_m_t192_s42 | 0.7936 | 0.75 | 0.8157 | 0.875 | 0.641 | 36 | YES |
| sf_m_t256_s42 | 0.8031 | 0.80 | 0.8231 | 0.884 | 0.645 | 31 | YES |
| sf_m_t320_s42 | 0.8021 | 0.70 | 0.8126 | 0.880 | 0.621 | 32 | YES |
| sf_mean_t192_s42 | 0.7837 | 0.85 | 0.8119 | 0.892 | 0.597 | 40 | no |
| sf_nonorm_t192_s42 | 0.7663 | 0.80 | 0.7765 | 0.904 | 0.496 | 54 | no |
| sf_noskip_t192_s42 | 0.7819 | 0.70 | 0.8052 | 0.854 | 0.655 | 43 | YES |
| sf_osr_t192_s42 | 0.7951 | 0.85 | 0.8218 | 0.889 | 0.632 | 57 | no |
| sf_pdk_t128_s42 | 0.7617 | 0.80 | 0.7811 | 0.878 | 0.542 | 53 | no |
| sf_pdk_t192_s42 | 0.7864 | 0.85 | 0.8115 | 0.871 | 0.637 | 52 | no |
| sf_pdkctl_t128_s42 | 0.7620 | 0.85 | 0.7815 | 0.853 | 0.586 | 52 | no |
| sf_s_d64_t256_s42 | 0.7990 | 0.85 | 0.8174 | 0.877 | 0.642 | 30 | YES |
| sf_sr244_t192_s42 | 0.7858 | 0.85 | 0.8112 | 0.865 | 0.649 | 49 | no |
| sf_sr444_t192_s42 | 0.7781 | 0.85 | 0.8002 | 0.878 | 0.590 | 20 | YES |
| sf_stemiso_t192_s42 | 0.7693 | 0.85 | 0.7959 | 0.877 | 0.582 | 56 | no |
| sf_wd1e2_t192_s42 | 0.7923 | 0.80 | 0.8063 | 0.896 | 0.577 | 46 | no |

## Gate 0: tile ladder + ERF law (prediction 2) — and C1's POWER CHECK

| tile | raw F0.5 |
|---|---|
| t64 | 0.6816 |
| t128 | 0.7578 |
| t192 | 0.7857 |
| t256 | 0.7948 |
| t320 | 0.8003 |

**G = raw(t192) − raw(t128) = +0.0279** (UNETR++ +0.055 · nnFormer +0.0063).
**C1/PDK 2x2 is POWERED.** Success needs PDK@t128 >= stock@t128 + 0.02 AND >= 50% of G (= +0.0139) AND >= control + 0.008.

## Gate 1: C1 / PDK — 2x2 with the mandatory same-cost control

| cell | raw |
|---|---|
| stock @t128 | 0.7578 |
| PDK @t128 | 0.7617 |
| PDK same-cost control @t128 | 0.7620 |

delta(PDK − stock) = **+0.0039** · delta(PDK − control) = **-0.0003** (the INFORMATION claim: the control has the same params, FLOPs and extra keys but no new pixels).

## Gate 2: C2 / DCM — delta=0 F0.5 is the SECONDARY readout

- DCM absolute: raw 0.7869 vs stock 0.7857 = **+0.0012** (inside the ±0.008 floor)
- DCM surface: raw 0.7851 vs stock 0.7857 = **-0.0006** (inside the ±0.008 floor)

Primary readouts live in the analysis job: the learned table peak vs the Inkalyzer profile, and the stress SIGN ASYMMETRY (absolute should lose >= 0.02 at |delta| >= 6; surface should not).

## Gate 3: C3 / OSR and C4 / DSM screens (pre-registered: both neutral)

- C3 OSR: 0.7951 vs 0.7857 = **+0.0094** → outside the floor — promote to 3 seeds + t256 confirmation
- C4 DSM: 0.7862 vs 0.7857 = **+0.0006** → NEUTRAL — informative null, no seeds spent
  - post-filter SUBSTITUTION (the primary readout): stock gain +0.0235 → DSM gain +0.0279 (no substitution; needs a >= 40% reduction at |delta raw| <= 0.008)

## Capacity x context grid (the capacity-confound-free ladder)

| capacity | t192 | t256 | t320 |
|---|---|---|---|
| SF-S 4.44M | 0.7857 | 0.7948 | 0.8003 |
| SF-M 19.97M | 0.7936 | 0.8031 | 0.8021 |
| SF-L 79.35M | — | — | — |

## Benchmark row (cross-model headline)

| model | raw | post |
|---|---|---|
| UNETR++ c1_t256 (3 seeds) | 0.8111 | 0.8190 |
| UNETR++ lean_t192_s42 | 0.7784 | 0.7957 |
| nnFormer nnf_c1_t256 (3 seeds) | 0.7801 | 0.8005 |
| nnFormer n_base_t192 (3 seeds) | 0.7657 | 0.7836 |
