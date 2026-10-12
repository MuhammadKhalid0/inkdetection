# Project description

**3D Transformers for Ink Detection in Carbonized Herculaneum Papyri — a controlled benchmark**

This MSc thesis project (Pattern Recognition Lab, FAU Erlangen-Nürnberg) studies how well modern
3D medical-imaging transformers detect ink in micro-CT scans of carbonized papyrus scrolls.
The input is a stack of surface layers sampled around a virtually unwrapped papyrus sheet
(Scroll 5, PHerc. 0172). The output is a 2D ink mask.

Three architectures are ported, adapted and evaluated under **one** fixed protocol (same data,
split, registered labels, training recipe, inference, post-processing, metrics and
interpretability probes). Only the backbone changes between them:

1. **UNETR++** (efficient paired attention): the reference model and the best result,
   **F0.5 = 0.819 ± 0.006** (3 seeds, post-processed) and 0.826 for the best single seed.
2. **nnFormer** (local windowed volumetric attention): **0.801 ± 0.010**.
3. **SegFormer3D** (sequence-reduced attention with an all-MLP decoder): **0.823** for the best
   single-seed cell. This is the lightest model and runs at 4–20 M parameters.

Each model also gets small, ablation-isolated contributions (surface-relative depth priors,
cross-scale skip attention, peripheral downsampled keys and others). Each one starts as an exact
no-op at initialisation, so every measured difference can be attributed to it.

The study finds that **context scale, evaluation protocol and label quality matter more than
architecture**. It also gives a mechanistic reason why local windowed attention cannot exploit
larger tiles.
