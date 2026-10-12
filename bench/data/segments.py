"""Canonical dataset paths and the fixed train/val/test split.

Split (authoritative, per Fei Wu 2026-07-01): all 9 labelled segments are from
Scroll 5 (PHerc. 0172, Unlost project).  Validation = 20241113090990,
Test = the auto_grown 'flatboi' inference segment (NO labels -> qualitative only).
Train = the other 8 labelled segments.
"""
import os

# --- filesystem roots -------------------------------------------------------
VAULT_ROOT = "/home/vault/iwi5/iwi5401h"
WORK_ROOT = os.environ.get("INK_WORK_ROOT", "/home/woody/iwi5/iwi5401h/vesuvius")

RAW_LABELED = os.path.join(VAULT_ROOT, "new_labeled")
RAW_INFER = os.path.join(VAULT_ROOT, "new_inference")

PROC_ROOT = os.path.join(WORK_ROOT, "data", "proc")          # processed labelled
PROC_INFER_ROOT = os.path.join(WORK_ROOT, "data", "proc_infer")
MANIFEST_ROOT = os.path.join(WORK_ROOT, "data", "manifests")
# INK_OUTPUT_ROOT (2026-07-17): new experiment rounds write to a fresh root
# (e.g. outputs2) so historical run folders are never touched. Default is the
# original path -> all old scripts/behaviour unchanged unless the env var is set.
OUTPUT_ROOT = os.environ.get("INK_OUTPUT_ROOT", os.path.join(WORK_ROOT, "outputs"))

# --- segments ---------------------------------------------------------------
SEGMENTS = [
    "20241025145341",
    "20241025145701",
    "20241030152031",
    "20241108111522",
    "20241108115232",
    "20241108120732",
    "20241113070770",
    "20241113080880",
    "20241113090990",
]

VAL_SEGMENT = "20241113090990"
TEST_SEGMENT = "20251115002740-auto_grown_20251115002740308_0_flatboi"
TRAIN_SEGMENTS = [s for s in SEGMENTS if s != VAL_SEGMENT]


def raw_vol_path(seg: str) -> str:
    return os.path.join(RAW_LABELED, seg, "layers", f"{seg}.npy")


def raw_ink_path(seg: str) -> str:
    return os.path.join(RAW_LABELED, seg, f"{seg}_inklabels.png")


def raw_mask_path(seg: str) -> str:
    return os.path.join(RAW_LABELED, seg, f"{seg}_mask.png")


def proc_dir(seg: str) -> str:
    return os.path.join(PROC_ROOT, seg)


# --- inference (test) segment: 65 raw tifs + tifxyz_mask.png, NO labels -----
def raw_infer_layers_dir(seg: str) -> str:
    return os.path.join(RAW_INFER, seg, "layers")


def raw_infer_mask_path(seg: str) -> str:
    return os.path.join(RAW_INFER, seg, "tifxyz_mask.png")


def proc_infer_dir(seg: str) -> str:
    return os.path.join(PROC_INFER_ROOT, seg)
