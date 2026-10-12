"""Master results table + PRE-REGISTERED gate verdicts (2026-07-20).

Collects every nnFormer run on disk and applies the plan's decision rules
mechanically, so the verdicts are not chosen after seeing the numbers:

  * noise floor: single-seed |delta| <= 0.008 -> "neutral / parked" (plan E4)
  * CSSA (plan 2.2): success = CSSA@t128 closes >= HALF the stock t128->t192 gap
    (>= +0.02 raw over stock@t128); deployment gate = +0.008 at t256
  * SR-PB (plan 2.1): delta=0 expected NEUTRAL; the real readout is the G1 stress
    tails at |delta| >= 4 (reported here if the stress json exists)
  * capacity/density (N2): overfit signature = best_epoch early AND train loss
    still falling after the val peak

Writes docs/RESULTS.md (thesis-ready tables) + docs/results.json.
"""
import csv
import glob
import json
import os

WORK = "/home/woody/iwi5/iwi5401h"
OUT = f"{WORK}/nnFormer/outputs"
DOCS = f"{WORK}/nnFormer/docs"
NOISE = 0.008

# UNETR++ reference numbers (frozen, ink_reg labels) for the cross-model row
UNETRPP = {"c1_t256 (3 seeds)": (0.8111, 0.8190),
           "t256_lean_s42": (0.8085, 0.8181),
           "lean_t192_s42": (0.7784, 0.7957),
           "r4 pipeline": (0.7909, 0.7986)}


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
            rec["loss_drop_after_best"] = round(loss[b] - loss[-1], 4)
            rec["val_drop_after_best"] = round(val[-1] - val[b], 4)
            rec["overfit_signature"] = bool(
                b < 0.75 * len(rows) and (loss[b] - loss[-1]) > 0.02
                and (val[-1] - val[b]) < -0.003)
    tta = glob.glob(os.path.join(d, "*tta*.json"))
    if tta:
        try:
            t = json.load(open(tta[0]))
            rec["tta"] = t
        except Exception:
            pass
    return rec


def fmt(v, n=4):
    return "—" if v is None else f"{v:.{n}f}"


def main():
    runs = sorted(os.path.basename(p) for p in glob.glob(os.path.join(OUT, "*"))
                  if os.path.isdir(p))
    recs = {r: load(r) for r in runs}
    recs = {k: v for k, v in recs.items() if v}

    L = ["# nnFormer — master results (auto-generated)", "",
         "All numbers: masked swept-F0.5 on val 20241113090990, tau selected on val,",
         "post = 8-connected min-size filter. Noise floor: single-seed |d| <= 0.008.", ""]

    L += ["## All runs", "",
          "| run | labels | raw | tau | post | P | R | best ep | overfit? |",
          "|---|---|---|---|---|---|---|---|---|"]
    for r, v in sorted(recs.items()):
        L.append(f"| {r} | {v.get('labels','?')} | {fmt(v.get('raw'))} | "
                 f"{fmt(v.get('tau'),2)} | {fmt(v.get('post'))} | {fmt(v.get('P'),3)} | "
                 f"{fmt(v.get('R'),3)} | {v.get('best_epoch','—')} | "
                 f"{'YES' if v.get('overfit_signature') else 'no'} |")

    # ---- gate 1: CSSA 2x2 (plan §2.2, Table N-V) -------------------------
    L += ["", "## Gate: CSSA (contribution C2) — pre-registered 2x2", ""]
    b128, c128 = recs.get("n_base_t128_r1_s42"), recs.get("n_cssa_t128_r1_s42")
    b192, c192 = recs.get("n_base_t192_r1_s42"), recs.get("n_cssa_t192_r1_s42")
    if all((b128, c128, b192, c192)):
        gap = b192["raw"] - b128["raw"]
        closed = c128["raw"] - b128["raw"]
        frac = closed / gap if gap > 1e-9 else float("nan")
        L += ["| | t128 | t192 | d(128->192) |", "|---|---|---|---|",
              f"| stock | {fmt(b128['raw'])} | {fmt(b192['raw'])} | {gap:+.4f} |",
              f"| CSSA | {fmt(c128['raw'])} | {fmt(c192['raw'])} | "
              f"{c192['raw']-c128['raw']:+.4f} |",
              f"| d(CSSA-stock) | {c128['raw']-b128['raw']:+.4f} | "
              f"{c192['raw']-b192['raw']:+.4f} | |", "",
              f"Stock t128->t192 gap = **{gap:+.4f}**; CSSA@t128 closes "
              f"**{closed:+.4f}** = **{frac*100:.0f}%** of it.", ""]
        if closed >= 0.02 and frac >= 0.5:
            L.append("**VERDICT: PASS** — pre-registered success criterion met "
                     "(>= half the gap). Promote all four cells to 3 seeds; "
                     "letterform context CAN be supplied architecturally.")
        else:
            L.append("**VERDICT: FAIL (informative null)** — the pre-written reading "
                     "applies: letterform context must enter at input resolution; it "
                     "cannot be retrieved from coarse features after the fact. This "
                     "*strengthens* the tile-ladder finding. Report the gamma "
                     "trajectory as mechanism evidence; do NOT spend promotion seeds.")
    else:
        L.append("_(2x2 incomplete — awaiting runs)_")

    # ---- gate 2: SR-PB (plan §2.1) — BAND vs BAND -------------------------
    # 2026-08-14 lesson: stock seed 42 was the WEAKEST stock seed (0.7581 vs
    # 0.7687/0.7703). A band-vs-single comparison inflated SR-PB to +0.013;
    # the pre-registered rule is band vs band, so that is what is reported.
    L += ["", "## Gate: SR-PB (contribution C1) — band vs band", ""]
    import statistics as st
    sb = [recs[r] for r in ("n_srpb_t192_r1_s42", "n_srpb_t192_r1_s43",
                            "n_srpb_t192_r1_s44") if r in recs]
    bb = [recs[r] for r in ("n_base_t192_r1_s42", "n_base_t192_r1_s43",
                            "n_base_t192_r1_s44") if r in recs]
    if sb and bb:
        def band(v, k):
            x = [r[k] for r in v if r.get(k) is not None]
            return (st.mean(x), st.stdev(x) if len(x) > 1 else 0.0, len(x))
        sr, ss, sn = band(sb, "raw"); br, bs, bn = band(bb, "raw")
        sp, sps, _ = band(sb, "post"); bp, bps, _ = band(bb, "post")
        d_raw, d_post = sr - br, sp - bp
        L += ["| | raw | post |", "|---|---|---|",
              f"| stock t192 (n={bn}) | {br:.4f} ± {bs:.4f} | {bp:.4f} ± {bps:.4f} |",
              f"| SR-PB t192 (n={sn}) | {sr:.4f} ± {ss:.4f} | {sp:.4f} ± {sps:.4f} |",
              f"| **delta (band vs band)** | **{d_raw:+.4f}** | **{d_post:+.4f}** |", ""]
        seeds = ", ".join(f"s{r['run'][-2:]} {r['raw']:.4f}" for r in bb)
        L.append(f"Stock seeds individually: {seeds}.")
        if abs(d_raw) <= NOISE and abs(d_post) <= NOISE:
            L.append(f"**VERDICT: NEUTRAL on F0.5** — both deltas inside the ±{NOISE} floor. "
                     "What survives: seed-variance reduction "
                     f"(raw sd {ss:.4f} vs {bs:.4f}) and the Inkalyzer depth-broadening "
                     "mechanism. The depth-offset stress curve still uses ONLY base s42 "
                     "(the weakest seed) as baseline — replicate on s43/s44 before claiming.")
        else:
            L.append(f"**VERDICT: outside the noise band** (raw {d_raw:+.4f}, post {d_post:+.4f}).")
        if bn < 3:
            L.append(f"_(stock band incomplete: n={bn} — do NOT claim until 3 seeds)_")
    else:
        L.append("_(SR-PB / stock bands incomplete)_")
    sj = glob.glob(f"{WORK}/nnFormer/outputs/zshift_stress/zshift_stress.json")
    if sj:
        L += ["", "Stress runs present: " +
              ", ".join(r["run"] for r in json.load(open(sj[0])))]

    # ---- benchmark row ----------------------------------------------------
    L += ["", "## Benchmark row (cross-model headline)", "",
          "| model | raw | post |", "|---|---|---|"]
    for k, (a, b) in UNETRPP.items():
        L.append(f"| UNETR++ {k} | {a:.4f} | {b:.4f} |")
    seeds = [recs[r] for r in ("nnf_s_d64_t256_s42", "nnf_s_d64_t256_s43",
                               "nnf_s_d64_t256_s44") if r in recs]
    if seeds:
        import statistics as st
        raws = [s["raw"] for s in seeds if s.get("raw")]
        posts = [s["post"] for s in seeds if s.get("post")]
        sd = (f" ± {st.stdev(raws):.4f}" if len(raws) > 1 else " (n=1)")
        sdp = (f" ± {st.stdev(posts):.4f}" if len(posts) > 1 else " (n=1)")
        L.append(f"| **nnFormer nnf_s_d64_t256 ({len(seeds)} seed(s))** | "
                 f"{st.mean(raws):.4f}{sd} | {st.mean(posts):.4f}{sdp} |")
        best = max(p for p in posts) if posts else None
        L += ["", f"Gap to UNETR++ c1_t256 (0.8190 post): "
                  f"**{(best - 0.8190):+.4f}** at best nnFormer seed." if best else ""]

    # ---- budget test ------------------------------------------------------
    e90, e60 = recs.get("nnf_s_d64_t256_e90_s42"), recs.get("nnf_s_d64_t256_s42")
    if e90 and e60:
        L += ["", "## Budget test (N2 flag: val still rising at e59)", "",
              f"e90: raw {fmt(e90['raw'])} (best ep {e90['best_epoch']}) vs "
              f"e60 same seed: raw {fmt(e60['raw'])} (best ep {e60['best_epoch']}) "
              f"-> **{e90['raw']-e60['raw']:+.4f}**",
              "", "UNETR++ precedent: e90 < e60 (0.8070 vs 0.8141) — the 60-epoch "
              "cosine was confirmed there. This tests whether nnFormer differs."]

    os.makedirs(DOCS, exist_ok=True)
    open(os.path.join(DOCS, "RESULTS.md"), "w").write("\n".join(L) + "\n")
    json.dump(recs, open(os.path.join(DOCS, "results.json"), "w"), indent=1)
    print("\n".join(L))
    print("\n->", os.path.join(DOCS, "RESULTS.md"))


if __name__ == "__main__":
    main()
