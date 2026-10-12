# Vesuvius Thesis — Verified Data Facts (checked 2026-06-24)

## Workspace
- 3 roots: /home/hpc/iwi5/iwi5401h (home, code+plans), /home/vault/iwi5/iwi5401h (Vault data), /scratch/iwi5401h (Scratch data, MIRROR of Vault).
- $WORK = /home/woody/iwi5/iwi5401h. Inspection venv at $WORK/inspect_env (numpy 2.0.2 + pillow).
- Plans in: plan(md_files)/ — implementation_guide, methodology_plan, onboarding_summary.
- No numpy in system python (3.9). Module python/3.12-base exists. No conda.

## Data layout (Vault == Scratch)
- new_labeled/<SEG>/  (9 segments)
    - layers/00.tif..64.tif (65 layers, uint8 'L'), plus layers/<SEG>.npy = pre-stacked vol
    - <SEG>_inklabels.png (RGBA, R=G=B grayscale, near-binary), <SEG>_mask.png (binary 0/255)
- new_inference/20251115002740-auto_grown_..._flatboi/ (1 segment, INFERENCE ONLY)
    - layers/00.tif..64.tif (65, uint8), tifxyz_mask.png. NO inklabels.

## Volume .npy
- shape (65, H, W), dtype uint8. Already 8-bit (max ~183 in samples; clipping likely already applied).
- Segment shapes (H,W): 145341=(1340,7620) 145701=(1340,7520) 152031=(10580,22000 HUGE)
  111522=(7420,4820) 115232=(7380,4400) 120732=(7440,4800) 070770=(1580,11360)
  080880=(1600,11400) 090990=(1620,11480)

## Labels
- Near-BINARY: ~0 or ~255, <1% antialiased mid values. Threshold @127 -> binary.
- Ink fraction per seg: 5.3%-15.8% (consistent w/ ~10% class imbalance).

## Alignment (MUST FIX in preprocessing)
- mask is 1px smaller in H and W than vol/ink for ALL segs.
- 20241108120732: vol H=7440 but ink H=7424 (16px mismatch). Crop all to common min H,W.

## Depth peaks (intensity argmax over depth, per seg)
- 27,42,28,28,30,29,30,31,29. Most ~27-31 (near center 32). 145701 peaks at 42 -> OFFSET OUTLIER, empirically confirms surface-offset problem.

## Inference segment
- 65 layers uint8, size W49240 x H11140 (~549 MP/layer, huge). mask only, no labels -> cross-scroll QUALITATIVE only.

## DIVERGENCES FROM PLAN (flag to user/supervisor)
1. 9 labeled segments, NOT 3 IR fragments. Timestamp-named auto-segmented (likely Scroll-1 GP pseudo/corrected labels), not Kaggle fragments.
2. Plan warns scoring vs pseudo-labels is circular -> MUST confirm with Fei Wu if these are trustworthy labels or pseudo.
3. No second-scroll labels -> cross-scroll regime is qualitative (1 inference seg).
4. Volume already uint8 pre-stacked -> prepare.py simpler.
5. LOFO becomes cross-segment within (likely) Scroll 1, not 3-fragment same-domain.
