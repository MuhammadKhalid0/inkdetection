"""Generate every SegFormer3D-Ink config from the plan (segformer3d_plan.md Appendix A).

Each cell is a one-key delta from sf_base_t192_s42, so the diff IS the ablation.
"""
import copy, os, yaml

SF = "/home/woody/iwi5/iwi5401h/segFormer"
OUT = f"{SF}/configs"
ROOT = f"{SF}/outputs"

MANIFEST = {64: "_reg", 128: "_reg_t128", 192: "_reg_t192", 256: "_reg_t256",
            320: "_reg_t320"}

BASE = {
    "model": {"name": "segformer3d", "feature_size": 16, "capacity": "S",
              "stem": "dp", "sr": "144_122", "final": "expand", "skip_stem": True,
              "drop_path": 0.0, "dw_norm": "bn", "dec_proj_ln": True, "dec_dim": 256,
              "scale_mode": "head_dim", "pdk": False, "dcm": False,
              "sr_overlap": False, "dsm": False},
    "shared": {"texture_stem": False, "collapse": "surface_relative"},
    "data": {"tile": 192, "depth": 32, "center_layer": 32, "n_layers": 65,
             "normalize": 255.0, "label_file": "ink_reg.npy",
             "manifest_suffix": "_reg_t192", "pos_thresh": 0.05,
             "samples_per_epoch": 4000, "use_uncertainty": True,
             "halo": {"enabled": False, "factor": 2},
             "aug": {"hflip": True, "vflip": True, "rot90": True, "z_flip": False,
                     "z_flip_p": 0.5, "z_shift": 2, "z_shift_p": 0.3}},
    "train": {"loss": "dice_bce_masked", "lr": 1.0e-4, "weight_decay": 3.0e-5,
              "epochs": 60, "warmup_epochs": 3, "batch": 4, "num_workers": 8,
              "amp": True, "amp_dtype": "bf16", "grad_clip": 1.0, "val_tau": 0.4,
              "select_on": "sweep", "sweep_every": 1, "patience": 20, "seed": 42},
    "infer": {"stride": 96, "batch": 4},
    "output": {"dir": ROOT, "run_name": "sf_base_t192_s42"},
}


def cfg(run, tile=192, seed=42, model=None, data=None, train=None, infer=None,
        manifest=None, batch=None):
    c = copy.deepcopy(BASE)
    c["data"]["tile"] = tile
    c["data"]["manifest_suffix"] = manifest or MANIFEST[tile]
    c["infer"]["stride"] = tile // 2
    if batch:
        c["train"]["batch"] = batch
    if model:
        c["model"].update(model)
    if data:
        for k, v in data.items():
            if isinstance(v, dict) and isinstance(c["data"].get(k), dict):
                c["data"][k].update(v)
            else:
                c["data"][k] = v
    if train:
        c["train"].update(train)
    if infer:
        c["infer"].update(infer)
    c["train"]["seed"] = seed
    c["output"]["run_name"] = run
    return run, c


# train.resume / train.time_limit_min (2026-09-12): the four cells that hit the 24 h A40
# wall rerun on A100 as resumable chains (scripts/slurm_chain_a100.sbatch). 1320 min =
# 22 h of training per 24 h chunk, leaving margin for smoke test and post-processing.
CELLS = [
    # ---- S0 sanity (not reported) ----------------------------------------------
    cfg("s0_sanity_t64", tile=64, batch=16,
        train={"epochs": 2, "warmup_epochs": 1}, data={"samples_per_epoch": 2000},
        infer={"stride": 32, "batch": 32}),

    # ---- S1 adaptation screens (t192, seed 42) — 12 cells -----------------------
    cfg("sf_base_t192_s42"),
    cfg("sf_stemiso_t192_s42", model={"stem": "iso"}),                        # A1
    cfg("sf_sr444_t192_s42", model={"sr": "444_222"}),                        # A2
    cfg("sf_sr244_t192_s42", model={"sr": "244_222"}),                        # A2
    cfg("sf_nonorm_t192_s42", model={"dw_norm": "none", "dec_proj_ln": False}),  # A3
    cfg("sf_noskip_t192_s42", model={"skip_stem": False}),                    # A4
    cfg("sf_finlow_t192_s42", model={"final": "trilinear_lowres_pred"}),      # A5
    cfg("sf_mean_t192_s42", model={}),                                        # A6 (below)
    cfg("sf_dec128_t192_s42", model={"dec_dim": 128}),                        # A7
    cfg("sf_deep_t192_s42", model={}),                                        # A8 (below)
    cfg("sf_lr5e5_t192_s42", train={"lr": 5.0e-5}),                           # A12
    cfg("sf_wd1e2_t192_s42", train={"weight_decay": 1.0e-2}),                 # A13

    # ---- S1b tile ladder + t320 (the power check, run BEFORE the C1 gate) -------
    cfg("sf_base_t64_s42", tile=64, batch=16),
    cfg("sf_base_t128_s42", tile=128, batch=4),
    cfg("sf_base_t256_s42", tile=256),
    cfg("sf_base_t320_s42", tile=320, train={"resume": True, "time_limit_min": 1320}),   # batch 4: keeps 1000 steps/ep; A100 chain
    cfg("sf_base_d96_t320_s42", tile=320, manifest="_reg_t320_d96", train={"resume": True, "time_limit_min": 1320}),

    # ---- S1c contribution screens ----------------------------------------------
    cfg("sf_pdk_t192_s42", model={"pdk": True, "pdk_stages": [2, 3, 4],
                                  "pdk_grid": [8, 8, 8]},
        data={"halo": {"enabled": True, "factor": 2}}),
    cfg("sf_pdk_t128_s42", tile=128, model={"pdk": True, "pdk_stages": [2, 3, 4],
                                            "pdk_grid": [8, 8, 8]},
        data={"halo": {"enabled": True, "factor": 2}}),
    cfg("sf_pdkctl_t128_s42", tile=128, model={"pdk": True, "pdk_stages": [2, 3, 4],
                                               "pdk_grid": [8, 8, 8]},
        data={"halo": {"enabled": True, "factor": 2, "mode": "self"}}),
    cfg("sf_dcm_abs_t192_s42", model={"dcm": True, "dcm_frame": "absolute"}),
    cfg("sf_dcm_surf_t192_s42", model={"dcm": True, "dcm_frame": "surface"}),
    cfg("sf_osr_t192_s42", model={"sr_overlap": True}),
    cfg("sf_dsm_t192_s42", model={"dsm": True, "dsm_dilation": 2}),

    # ---- S2 capacity x context --------------------------------------------------
    cfg("sf_m_t192_s42", model={"capacity": "M"}),
    cfg("sf_m_t256_s42", tile=256, model={"capacity": "M"}, train={"resume": True, "time_limit_min": 1320}),
    cfg("sf_m_t320_s42", tile=320, model={"capacity": "M"}, train={"resume": True, "time_limit_min": 1320}),
    cfg("sf_l_t256_s42", tile=256, model={"capacity": "L"}),
    cfg("sf_s_d64_t256_s42", tile=256, manifest="_reg_t256_d64"),

    # ---- A10 prior-art control: 2.5D depth-as-channels (NOT a benchmark row) -----
    cfg("sf_25d_t256_s42", tile=256, model={"mode": "25d"}),

    # ---- A15 depth-window ablation: the FULL 65-layer stack (2026-09-24) ---------
    # Every other cell feeds the 32-layer window [16,48) that chapter 1 fixed. This one
    # feeds all 65 layers at the reference tile, so the only change vs sf_base_t192_s42 is
    # how much of the depth axis the model sees. The backbone is parameter-identical
    # (4,437,568); only the shared collapse's relative-offset table grows (63 -> 129).
    # Note: with depth == n_layers the z_shift augmentation has no room and is inert.
    cfg("sf_depth65_t192_s42", tile=192,
        data={"depth": 65},
        train={"resume": True, "time_limit_min": 1320}),
]

# A6 / A8 need non-model keys
FIX = {"sf_mean_t192_s42": ("shared", {"collapse": "mean"}),
       "sf_deep_t192_s42": ("model", {"depths": [2, 2, 5, 2]})}

if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    for run, c in CELLS:
        if run in FIX:
            sec, upd = FIX[run]
            c[sec].update(upd)
        with open(os.path.join(OUT, f"{run}.yaml"), "w") as f:
            yaml.safe_dump(c, f, sort_keys=False, default_flow_style=False)
    print(f"wrote {len(CELLS)} configs to {OUT}")
    for run, _ in CELLS:
        print("  ", run)
