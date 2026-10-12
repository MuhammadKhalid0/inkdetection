"""Decision tests for scripts/chain_guard.py on fabricated run directories (CPU, seconds)."""
import json, os, subprocess, sys, tempfile
import torch

G = [sys.executable, "/home/woody/iwi5/iwi5401h/segFormer/scripts/chain_guard.py"]
PASS, FAIL = [], []


def check(name, cond, extra=""):
    (PASS if cond else FAIL).append(name)
    print(f"  [{'ok ' if cond else 'FAIL'}] {name}{(' -- ' + extra) if extra else ''}")


def run(*args, env=None):
    r = subprocess.run(G + list(args), capture_output=True, text=True,
                       env={**os.environ, "SLURM_JOB_ID": "999", **(env or {})})
    return r.returncode, r.stdout.strip()


def fresh():
    return tempfile.mkdtemp(prefix="guard_")


def last(d, epoch):
    torch.save({"epoch": epoch}, os.path.join(d, "last.pt"))


d = fresh()
check("empty dir -> train", run("decide", "--out", d)[1] == "train")

run("begin", "--out", d)                                   # chunk 1 starts at epoch 0
last(d, 20)                                                # chunk 1 got to epoch 20
check("progress made -> train", run("decide", "--out", d)[1] == "train")

run("begin", "--out", d)                                   # chunk 2 starts at 21
check("no progress in a whole chunk -> stop", run("decide", "--out", d)[1].startswith("stop:no_progress"),
      run("decide", "--out", d)[1])

d = fresh()
for i in range(6):
    last(d, i * 5); run("begin", "--out", d)
last(d, 40)
check("runaway guard -> stop at max_chunks",
      run("decide", "--out", d, "--max-chunks", "6")[1].startswith("stop:max_chunks"))

d = fresh(); last(d, 10); run("begin", "--out", d); run("crash", "--out", d, "--rc", "137")
check("crashed chain -> stop, never retries", run("decide", "--out", d)[1].startswith("stop:crashed"),
      run("decide", "--out", d)[1])

d = fresh(); last(d, 59); json.dump({"reason": "completed"}, open(os.path.join(d, "train_done.json"), "w"))
check("trained, not post-processed -> post", run("decide", "--out", d)[1] == "post")
open(os.path.join(d, "postprocess.json"), "w").write("{}")
check("trained and post-processed -> stop:complete", run("decide", "--out", d)[1] == "stop:complete")

d = fresh()
json.dump({"best_epoch": 12}, open(os.path.join(d, "summary.json"), "w"))
torch.save({"epoch": 12}, os.path.join(d, "best.pt"))
check("consistent best.pt/summary -> exit 0", run("consistent", "--out", d)[0] == 0)
torch.save({"epoch": 13}, os.path.join(d, "best.pt"))
check("mismatched best.pt/summary -> exit 1", run("consistent", "--out", d)[0] == 1)

d = fresh(); files_before = set(os.listdir(d)); last(d, 3)
for cmd in (["decide"], ["begin"], ["status"], ["crash", "--rc", "1"]):
    run(*cmd, "--out", d)
check("guard never deletes a training artifact", os.path.exists(os.path.join(d, "last.pt")))

print(f"\n{len(PASS)} passed, {len(FAIL)} failed")
sys.exit(1 if FAIL else 0)
