"""Decision logic for chained (resumable) training jobs. Pure bookkeeping: this script
never deletes, moves or overwrites any training artifact.

  decide  --out DIR --max-chunks N   -> prints one of: train | post | stop:<reason>
  begin   --out DIR                  -> records that a training chunk is starting
  crash   --out DIR --rc RC          -> marks the chain as crashed (stops all continuations)
  consistent --out DIR               -> exit 1 if best.pt and summary.json disagree
  status  --out DIR                  -> human-readable summary

Stop reasons: crashed (a chunk failed; needs a human), complete (trained + post-processed),
max_chunks (runaway guard), no_progress (a whole chunk advanced zero epochs).
"""
import argparse, json, os, sys, time

STATE = "chain_state.json"


def _load(out):
    p = os.path.join(out, STATE)
    return json.load(open(p)) if os.path.exists(p) else {"chunks": 0, "starts": [], "jobs": []}


def _save(out, st):
    p = os.path.join(out, STATE)
    tmp = p + ".tmp"
    with open(tmp, "w") as f:
        json.dump(st, f, indent=2)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, p)


def _next_start_epoch(out):
    """Epoch a resumed trainer would start at (0 when there is no last.pt)."""
    p = os.path.join(out, "last.pt")
    if not os.path.exists(p):
        return 0
    import torch
    return int(torch.load(p, map_location="cpu", weights_only=False)["epoch"]) + 1


def decide(out, max_chunks):
    st = _load(out)
    if st.get("crashed"):
        return f"stop:crashed(rc={st.get('rc')},job={st.get('crash_job')})"
    if os.path.exists(os.path.join(out, "train_done.json")):
        return "stop:complete" if os.path.exists(os.path.join(out, "postprocess.json")) else "post"
    if st["chunks"] >= max_chunks:
        return f"stop:max_chunks({st['chunks']})"
    nxt = _next_start_epoch(out)
    if st["starts"] and nxt <= st["starts"][-1]:
        return f"stop:no_progress(previous chunk started at epoch {st['starts'][-1]}, still {nxt})"
    return "train"


def begin(out):
    st = _load(out)
    st["chunks"] += 1
    st["starts"].append(_next_start_epoch(out))
    st["jobs"].append({"job": os.environ.get("SLURM_JOB_ID"), "node": os.uname().nodename,
                       "started_at": time.strftime("%Y-%m-%d %H:%M:%S")})
    _save(out, st)
    print(f"chunk {st['chunks']} starting at epoch {st['starts'][-1]}")


def crash(out, rc):
    st = _load(out)
    st.update(crashed=True, rc=rc, crash_job=os.environ.get("SLURM_JOB_ID"),
              crashed_at=time.strftime("%Y-%m-%d %H:%M:%S"))
    _save(out, st)
    print(f"marked crashed (rc={rc}); continuations will stop. Artifacts left untouched.")


def consistent(out):
    import torch
    s = json.load(open(os.path.join(out, "summary.json")))
    b = torch.load(os.path.join(out, "best.pt"), map_location="cpu", weights_only=False)
    ok = int(b["epoch"]) == int(s["best_epoch"])
    print(f"best.pt epoch {b['epoch']} vs summary.json best_epoch {s['best_epoch']}: "
          f"{'consistent' if ok else 'MISMATCH — inspect before post-processing'}")
    return ok


def status(out):
    st = _load(out)
    done = os.path.exists(os.path.join(out, "train_done.json"))
    print(f"chunks={st['chunks']} starts={st['starts']} crashed={st.get('crashed', False)} "
          f"train_done={done} postprocess={os.path.exists(os.path.join(out, 'postprocess.json'))}")
    for j in st["jobs"]:
        print(f"  job {j['job']} on {j['node']} at {j['started_at']}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["decide", "begin", "crash", "consistent", "status"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-chunks", type=int, default=6)
    ap.add_argument("--rc", type=int, default=-1)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    if a.cmd == "decide":
        print(decide(a.out, a.max_chunks))
    elif a.cmd == "begin":
        begin(a.out)
    elif a.cmd == "crash":
        crash(a.out, a.rc)
    elif a.cmd == "consistent":
        sys.exit(0 if consistent(a.out) else 1)
    else:
        status(a.out)
