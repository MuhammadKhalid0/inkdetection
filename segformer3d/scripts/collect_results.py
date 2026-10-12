"""Master results table + PRE-REGISTERED gate verdicts for the SegFormer3D chapter.

Applies the plan's decision rules mechanically so verdicts are not chosen after seeing
the numbers.  BAND-VS-BAND FROM DAY 1: chapter 2's seed-42 trap (a contribution measured
against a single weak baseline seed read +0.013 and collapsed to +0.0065 once the
baseline had its own band) is encoded here, not left to the analyst's memory.

Writes docs/RESULTS.md + docs/results.json.
"""
import csv, glob, json, os, statistics as st

SF = "/home/woody/iwi5/iwi5401h/segFormer"
OUT, DOCS = f"{SF}/outputs", f"{SF}/docs"
NOISE = 0.008

REFERENCE = {                      # frozen numbers from the two finished chapters
    "UNETR++ c1_t256 (3 seeds)": (0.8111, 0.8190),
    "UNETR++ lean_t192_s42": (0.7784, 0.7957),
    "nnFormer nnf_c1_t256 (3 seeds)": (0.7801, 0.8005),
    "nnFormer n_base_t192 (3 seeds)": (0.7657, 0.7836),
}


def load(run):
    d = os.path.join(OUT, run)
    sp = os.path.join(d, "summary.json")
    if not os.path.exists(sp):
        return None
    s = json.load(open(sp))
    rec = {"run": run, "raw": s.get("best_f05"), "tau": s.get("best_tau"),
           "best_epoch": s.get("best_epoch"), "iou_ink": s.get("iou_ink"),
           "labels": s.get("label_file")}
    pp = os.path.join(d, "postprocess.json")
    if os.path.exists(pp):
        p = json.load(open(pp))["postprocessed"]
        rec.update(post=p.get("f05"), P=p.get("precision"), R=p.get("recall"))
    tl = os.path.join(d, "train_log.csv")
    if os.path.exists(tl):
        rows = list(csv.DictReader(open(tl)))
        if rows:
            loss = [float(r["train_loss"]) for r in rows]
            val = [float(r["val_f05_sweep"]) for r in rows]
            b = max(range(len(val)), key=lambda i: val[i])
            rec["n_epochs"] = len(rows)
            rec["overfit_signature"] = bool(
                b < 0.75 * len(rows) and (loss[b] - loss[-1]) > 0.02
                and (val[-1] - val[b]) < -0.003)
    return rec


def fmt(v, n=4):
    return "—" if v is None else f"{v:.{n}f}"


def band(recs, names, key="raw"):
    xs = [recs[n][key] for n in names if n in recs and recs[n].get(key) is not None]
    if not xs:
        return None
    return (st.mean(xs), st.stdev(xs) if len(xs) > 1 else 0.0, len(xs))


def main():
    runs = sorted(os.path.basename(p) for p in glob.glob(os.path.join(OUT, "*"))
                  if os.path.isdir(p))
    recs = {r: v for r, v in ((r, load(r)) for r in runs) if v}

    L = ["# SegFormer3D — master results (auto-generated)", "",
         "Masked swept-F0.5 on val 20241113090990 vs `ink_reg.npy`; tau selected on val;",
         "post = 8-connected min-size filter. Noise floor: single-seed |d| <= 0.008.", "",
         "## All runs", "",
         "| run | raw | tau | post | P | R | best ep | overfit? |",
         "|---|---|---|---|---|---|---|---|"]
    for r, v in sorted(recs.items()):
        L.append(f"| {r} | {fmt(v.get('raw'))} | {fmt(v.get('tau'),2)} | "
                 f"{fmt(v.get('post'))} | {fmt(v.get('P'),3)} | {fmt(v.get('R'),3)} | "
                 f"{v.get('best_epoch','—')} | {'YES' if v.get('overfit_signature') else 'no'} |")

    # ---- gate 0: the tile ladder AND the C1 power check (plan rule 3) -----------
    L += ["", "## Gate 0: tile ladder + ERF law (prediction 2) — and C1's POWER CHECK", ""]
    lad = [("t64", "sf_base_t64_s42"), ("t128", "sf_base_t128_s42"),
           ("t192", "sf_base_t192_s42"), ("t256", "sf_base_t256_s42"),
           ("t320", "sf_base_t320_s42")]
    have = [(n, recs[r]["raw"]) for n, r in lad if r in recs and recs[r].get("raw")]
    if have:
        L += ["| tile | raw F0.5 |", "|---|---|"] + \
             [f"| {n} | {v:.4f} |" for n, v in have] + [""]
        d = dict(have)
        if "t128" in d and "t192" in d:
            G = d["t192"] - d["t128"]
            L.append(f"**G = raw(t192) − raw(t128) = {G:+.4f}** "
                     f"(UNETR++ +0.055 · nnFormer +0.0063).")
            if G < 0.02:
                L.append("**C1/PDK 2x2 is UNDERPOWERED BY CONSTRUCTION — do not run it.** "
                         "The pre-registered rule (plan 2.1 gate 0) fires: evaluate PDK at "
                         "deployment scale only, and report the flat ladder as the headline "
                         "(it refutes prediction 2 and the ERF law).")
            else:
                L.append(f"**C1/PDK 2x2 is POWERED.** Success needs PDK@t128 >= "
                         f"stock@t128 + 0.02 AND >= 50% of G (= {0.5*G:+.4f}) AND "
                         f">= control + 0.008.")
    else:
        L.append("_(ladder incomplete)_")

    # ---- gate 1: C1 PDK 2x2 + same-cost control --------------------------------
    L += ["", "## Gate 1: C1 / PDK — 2x2 with the mandatory same-cost control", ""]
    b128, p128, c128 = (recs.get("sf_base_t128_s42"), recs.get("sf_pdk_t128_s42"),
                        recs.get("sf_pdkctl_t128_s42"))
    if all((b128, p128, c128)):
        L += ["| cell | raw |", "|---|---|",
              f"| stock @t128 | {fmt(b128['raw'])} |",
              f"| PDK @t128 | {fmt(p128['raw'])} |",
              f"| PDK same-cost control @t128 | {fmt(c128['raw'])} |", "",
              f"delta(PDK − stock) = **{p128['raw']-b128['raw']:+.4f}** · "
              f"delta(PDK − control) = **{p128['raw']-c128['raw']:+.4f}** "
              f"(the INFORMATION claim: the control has the same params, FLOPs and "
              f"extra keys but no new pixels)."]
    else:
        L.append("_(2x2 incomplete)_")

    # ---- gate 2: C2 DCM (F0.5 is NOT the primary readout) ----------------------
    L += ["", "## Gate 2: C2 / DCM — delta=0 F0.5 is the SECONDARY readout", ""]
    base, da, dsu = (recs.get("sf_base_t192_s42"), recs.get("sf_dcm_abs_t192_s42"),
                     recs.get("sf_dcm_surf_t192_s42"))
    if base and (da or dsu):
        for nm, r in (("absolute", da), ("surface", dsu)):
            if r:
                d = r["raw"] - base["raw"]
                L.append(f"- DCM {nm}: raw {r['raw']:.4f} vs stock {base['raw']:.4f} "
                         f"= **{d:+.4f}** ({'inside' if abs(d) <= NOISE else 'OUTSIDE'} "
                         f"the ±{NOISE} floor)")
        L.append("")
        L.append("Primary readouts live in the analysis job: the learned table peak vs "
                 "the Inkalyzer profile, and the stress SIGN ASYMMETRY (absolute should "
                 "lose >= 0.02 at |delta| >= 6; surface should not).")
    else:
        L.append("_(DCM cells incomplete)_")

    # ---- gate 3: C3 / C4 screens ------------------------------------------------
    L += ["", "## Gate 3: C3 / OSR and C4 / DSM screens (pre-registered: both neutral)", ""]
    for nm, run in (("C3 OSR", "sf_osr_t192_s42"), ("C4 DSM", "sf_dsm_t192_s42")):
        r = recs.get(run)
        if base and r:
            d = r["raw"] - base["raw"]
            v = ("NEUTRAL — informative null, no seeds spent" if abs(d) <= NOISE
                 else "outside the floor — promote to 3 seeds + t256 confirmation")
            L.append(f"- {nm}: {r['raw']:.4f} vs {base['raw']:.4f} = **{d:+.4f}** → {v}")
            if nm == "C4 DSM" and r.get("post") is not None:
                gs = base.get("post", 0) - base["raw"]
                gd = r["post"] - r["raw"]
                L.append(f"  - post-filter SUBSTITUTION (the primary readout): stock gain "
                         f"{gs:+.4f} → DSM gain {gd:+.4f} "
                         f"({'PASS' if gs > 0 and gd <= 0.6 * gs else 'no substitution'}; "
                         f"needs a >= 40% reduction at |delta raw| <= 0.008)")

    # ---- capacity x context grid ------------------------------------------------
    L += ["", "## Capacity x context grid (the capacity-confound-free ladder)", "",
          "| capacity | t192 | t256 | t320 |", "|---|---|---|---|"]
    for cap, pre in (("SF-S 4.44M", "sf_base"), ("SF-M 19.97M", "sf_m"),
                     ("SF-L 79.35M", "sf_l")):
        cells = []
        for t in (192, 256, 320):
            r = recs.get(f"{pre}_t{t}_s42")
            cells.append(fmt(r["raw"]) if r and r.get("raw") else "—")
        L.append(f"| {cap} | {cells[0]} | {cells[1]} | {cells[2]} |")

    # ---- benchmark row ----------------------------------------------------------
    L += ["", "## Benchmark row (cross-model headline)", "", "| model | raw | post |",
          "|---|---|---|"]
    for k, (a, b) in REFERENCE.items():
        L.append(f"| {k} | {a:.4f} | {b:.4f} |")
    seeds = [f"sf_c1_t256_s{s}" for s in (42, 43, 44)]
    br, bp = band(recs, seeds, "raw"), band(recs, seeds, "post")
    if br:
        sd = f" ± {br[1]:.4f}" if br[2] > 1 else " (n=1)"
        sp = f" ± {bp[1]:.4f}" if bp and bp[2] > 1 else " (n=1)"
        L.append(f"| **SegFormer3D sf_c1_t256 ({br[2]} seed(s))** | {br[0]:.4f}{sd} | "
                 f"{bp[0]:.4f}{sp} |" if bp else "")
        if bp:
            L += ["", f"Gap to UNETR++ c1_t256 (0.8190 post): **{bp[0]-0.8190:+.4f}** · "
                      f"to nnFormer (0.8005): **{bp[0]-0.8005:+.4f}**"]

    os.makedirs(DOCS, exist_ok=True)
    open(os.path.join(DOCS, "RESULTS.md"), "w").write("\n".join(L) + "\n")
    json.dump(recs, open(os.path.join(DOCS, "results.json"), "w"), indent=1)
    print("\n".join(L))
    print("\n->", os.path.join(DOCS, "RESULTS.md"))


if __name__ == "__main__":
    main()
