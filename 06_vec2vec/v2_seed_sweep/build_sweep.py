import nbformat as nbf
from pathlib import Path

HERE = Path(__file__).parent
cells = []
md = lambda s: cells.append(nbf.v4.new_markdown_cell(s.strip()))
code = lambda s: cells.append(nbf.v4.new_code_cell(s.strip()))

md("""
# Phase V2: vec2vec seed and data sweep (two runs at once, one per T4)

Neural Nexus, vector-DB compression project (team leader: Aman).

| | |
|---|---|
| **Time** | about 11 hours (10.5 h of training + setup). Kaggle's limit is 12 h |
| **Settings** | Accelerator = **GPU T4 x2**, Internet = **On**, Persistence = off |
| **Inputs** | FRIEND 1-4: Add Input -> Notebooks -> `phaseA_fp32_embeddings` and `phaseA2b_verify_attacker_extra` (Aman shares both with you). FRIEND 5: no inputs. FRIEND 6-7: only `phaseA_fp32_embeddings` |
| **The only line you change** | `FRIEND = ...` in cell 2 (Aman tells you your number) |
| **How to run** | first `QUICK_TEST = True` + Run All (about 10 min, checks everything works); then `QUICK_TEST = False` and **Save Version -> Save & Run All (Commit)**; close the tab |
| **Output** | `runs/<name>/`: log, curve.png, summary.json, resume.pt (to continue the run on the workstation), translators |

What it does: trains the authors' official vec2vec (commit 5efb3c0, their settings) twice at the same time,
one run per GPU. Each run differs only in its seed (the GAN's random starting point) or in its data.
The authors warn that their GAN often fails to start depending on the seed; this sweep looks for a seed and
data setting where it does start. Early stopping is OFF (in our first run the authors' rule stopped training
after about 6% of the training the paper uses). Every 30 minutes a full checkpoint is saved.

When it is finished: Share -> add `amanpriyansh` as **Can view**, and tell Aman.
""")

code(r"""
# Cell 1 - install + official vec2vec code at a fixed commit (2-4 minutes)
!mkdir -p /kaggle/working/code /kaggle/working/runs
!pip install -q "transformers==4.46.3" "vec2text==0.0.13" "sentence-transformers==3.4.1" accelerate toml wandb evaluate diffusers nltk seaborn
!rm -rf /kaggle/working/vec2vec && git clone -q https://github.com/rjha18/vec2vec.git /kaggle/working/vec2vec && cd /kaggle/working/vec2vec && git checkout -q 5efb3c02ee908847581c6488cf9c8ed6dace3c29 && git log -1 --format='vec2vec commit %H %cd'
""")

code(r"""
# Cell 2 - settings. CHANGE ONLY THESE TWO LINES.
FRIEND = 6            # 1-7 (Aman tells you which). 6 and 7 = same-source attacker runs (8 Nov)
QUICK_TEST = False    # True = 10-minute check on a tiny subset; False = the real 10.5-hour run

# The sweep. Each friend runs two lines, one per GPU. (The workstation runs ours fp32 seeds 7 and 8.)
SWEEP = {
    1: [dict(kind="ours", seed=5), dict(kind="ours", seed=1)],
    2: [dict(kind="ours", seed=2), dict(kind="ours", seed=3)],
    3: [dict(kind="ours", seed=4), dict(kind="ours", seed=6)],
    4: [dict(kind="dpr_matched", seed=5), dict(kind="dpr_matched", seed=1)],
    5: [dict(kind="native", seed=5, num_points=1_000_000), dict(kind="native", seed=5, num_points=50_000)],
    6: [dict(kind="ours_ss", seed=3), dict(kind="ours_ss", seed=4)],   # added 8 Nov: same-source attacker set,
    7: [dict(kind="ours_ss", seed=6), dict(kind="ours_ss", seed=7)],   # near-copies removed (workstation: seeds 1, 2, 5)
}
BUDGET_HOURS = {"ours": 10.5, "ours_ss": 10.5, "dpr_matched": 10.5, "native": 10.0}   # native epochs are long; keep room under 12 h

import os, sys, json, time, glob, subprocess
import numpy as np, torch
assert torch.cuda.device_count() == 2, "Set Accelerator = GPU T4 x2 (two GPUs), then restart"
RUNS = []
for spec in SWEEP[FRIEND]:
    name = spec["kind"] + (f"{spec['num_points'] // 1000}k" if spec["kind"] == "native" else "")
    name += f"_fp32_s{spec['seed']}" if spec["kind"] in ("ours", "ours_ss") else f"_s{spec['seed']}"
    RUNS.append({**spec, "name": name + ("_quicktest" if QUICK_TEST else "")})
for gpu, r in enumerate(RUNS):
    print(f"GPU {gpu}: {r['name']}   budget {0.05 if QUICK_TEST else BUDGET_HOURS[r['kind']]} h")
""")

code(r"""
# Cell 3 - what we expect BEFORE the run (written down so the result cannot move the goalposts)
EXPECT = {
    "random guess": "top-1 = 1/8,192 = 0.00012, average rank = 4,096 (ours / dpr_matched); "
                    "authors' top-1 within 1,024 = 0.001 (native)",
    "steps per epoch": "ours 195, dpr_matched 390, native-1M 3,906, native-50k 195 (batch 256)",
    "minutes per epoch on a T4": "ours about 0.7-2 (unknown before this run; the workstation needed 0.66 with two "
                                 "runs sharing a faster GPU); native much longer because it also runs two text encoders",
    "training steps in 10.5 h": "ours roughly 60,000-180,000. Our first run had only 19,000. The paper reports seeds "
                                "that converged within 10 epochs of 1M points, about 39,000 steps",
    "LEARNED (ours, dpr_matched)": "top-1 >= 0.01 (80x random) OR average rank <= a quarter of random (1,024 of 8,192) at any epoch",
    "LEARNED (native)": "authors' validation top-1 (within 1,024) >= 0.1 at any epoch",
    "chance that at least one of our 8 seeds learns (6 here + 2 on the workstation)": "unknown. If each seed had the paper's cross-backbone rate "
        "(e5->gtr: 3 of 15 seeds), about 1 - 0.8^8 = 0.83. The paper also reports gte->gtr failing without "
        "per-pair tuning, so zero is a real possibility and would itself be a result",
    "native-1M (positive control)": "should learn: it is the authors' own pair and pipeline. If it does not, the budget "
                                    "per run is too short and a failure of our pair is inconclusive",
    "native-50k vs native-1M": "if 1M learns and 50k does not, too little data is the problem (Aman's hypothesis)",
    "dpr_matched vs ours": "if dpr_matched learns and ours does not, more / matched victim-side data helps",
}
for k, v in EXPECT.items(): print(f"- {k}: {v}")
""")

code("%%writefile /kaggle/working/code/v2v_runner.py\n" + (HERE / "v2v_runner.py").read_text())
code("%%writefile /kaggle/working/code/v2v_job.py\n" + (HERE / "v2v_job.py").read_text())

code(r"""
# Cell 6 - check inputs and download the models / texts once (so the two runs do not download at the same time)
os.makedirs("/kaggle/working/runs", exist_ok=True)
kinds = {r["kind"] for r in RUNS}
if kinds & {"ours", "ours_ss", "dpr_matched"}:
    need = ["bge-large_victim_L32.npy", "gtr_victim_L32.npy"] + \
           (["gtr_attextra_L32.npy"] if kinds & {"ours", "dpr_matched"} else ["gtr_att_L32.npy"]) + \
           (["texts_extra.json"] if "dpr_matched" in kinds else [])
    for f in need:
        hits = glob.glob(f"/kaggle/input/**/{f}", recursive=True)
        assert hits, f"missing input {f}: Add Input -> Notebooks -> phaseA_fp32_embeddings and phaseA2b_verify_attacker_extra"
        print("found", hits[0])
if "dpr_matched" in kinds:
    from huggingface_hub import snapshot_download
    snapshot_download("BAAI/bge-large-en-v1.5")
if "native" in kinds:
    from datasets import load_dataset
    from sentence_transformers import SentenceTransformer
    t = time.time()
    ds = load_dataset("jxm/nq_corpus_dpr", split="train"); print(f"NQ DPR corpus: {len(ds):,} passages ({time.time()-t:.0f}s)")
    assert len(ds) >= 2 * 1_000_000 + 4096
    for m in ["infgrad/stella-base-en-v2", "thenlper/gte-base"]:
        SentenceTransformer(m, trust_remote_code=True); print("downloaded", m)
    del ds
""")

code(r"""
# Cell 7 - start both runs (one per GPU) and print progress every 10 minutes. This is the long cell.
procs = []
for gpu, r in enumerate(RUNS):
    out = f"/kaggle/working/runs/{r['name']}"
    os.makedirs(out, exist_ok=True)
    cmd = [sys.executable, "-u", "/kaggle/working/code/v2v_job.py", "--kind", r["kind"], "--seed", str(r["seed"]),
           "--hours", str(0.05 if QUICK_TEST else BUDGET_HOURS[r["kind"]]), "--data", "/kaggle/input",
           "--out", out, "--repo", "/kaggle/working/vec2vec"]
    if r["kind"] == "native": cmd += ["--num_points", str(r["num_points"])]
    if QUICK_TEST: cmd += ["--smoke"]
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu)}
    procs.append((r, out, subprocess.Popen(cmd, env=env, stdout=open(f"{out}/stdout.txt", "w"), stderr=subprocess.STDOUT)))
    time.sleep(60)   # stagger the starts

def last_line(out):
    try:
        rows = [json.loads(l) for l in open(f"{out}/v2v_log.jsonl")]
    except FileNotFoundError:
        return "starting"
    ep = [x for x in rows if x["kind"] == "ours"]
    if not ep: return "setting up / first epoch"
    e = ep[-1]
    if "top1" in e:
        return (f"epoch {e['epoch']}  {e['train_hours']:.2f} h  top1 {e['top1']:.4f}  avg_rank {e['avg_rank']:.0f}"
                f"  (best rank so far {min(x['avg_rank'] for x in ep):.0f})")
    val = [x for x in rows if x["kind"] == "authors_val"]
    t1 = [v for k, v in val[-1].items() if k.startswith("val/stella_gte_top_1_acc")] if val else []
    return f"epoch {e['epoch']}  {e['train_hours']:.2f} h  authors' top-1 {t1[0] if t1 else float('nan'):.4f}"

t0 = time.time()
while any(p.poll() is None for _, _, p in procs):
    time.sleep(30 if QUICK_TEST else 600)
    print(f"--- {(time.time()-t0)/3600:.2f} h since start")
    for r, out, p in procs:
        state = "running" if p.poll() is None else f"ended (code {p.returncode})"
        print(f"  {r['name']:24s} {state:16s} {last_line(out)}")
for r, out, p in procs:
    if p.returncode != 0:
        why = " (killed from outside: usually out of memory)" if p.returncode in (-9, 137) else ""
        print(f"\n!!! {r['name']} FAILED with exit code {p.returncode}{why}. Last lines of its stdout.txt:")
        print("".join(open(f"{out}/stdout.txt").readlines()[-30:]))
""")

code(r"""
# Cell 8 - results: verdict per run (rule fixed in cell 3) and the curves
from IPython.display import Image, display
rows = []
for r, out, p in procs:
    if not os.path.exists(f"{out}/summary.json"):
        rows.append({"run": r["name"], "status": "FAILED, see stdout.txt"}); continue
    s = json.load(open(f"{out}/summary.json"))
    if r["kind"] == "native":
        learned = (s["authors_val_top1_max"] or 0) >= 0.1
        rows.append({"run": r["name"], "hours": round(s["train_hours"], 2), "epochs": s["epochs"], "steps": s["steps"],
                     "authors_top1_last": s["authors_val_top1_last"], "authors_top1_max": s["authors_val_top1_max"],
                     "LEARNED": learned})
    else:
        learned = s["max_top1_any_epoch"] >= 0.01 or s["min_avg_rank_any_epoch"] <= (s["n_eval"] + 1) / 2 / 4
        rows.append({"run": r["name"], "hours": round(s["train_hours"], 2), "epochs": s["epochs"], "steps": s["steps"],
                     "min_per_epoch": round(s["minutes_per_epoch_median"], 2),
                     "top1_final": s["final"]["top1"], "avg_rank_final": round(s["final"]["avg_rank"]),
                     "top1_max": s["max_top1_any_epoch"], "avg_rank_min": round(s["min_avg_rank_any_epoch"]),
                     "LEARNED": learned})
import pandas as pd
table = pd.DataFrame(rows); print(table.to_string(index=False))
json.dump({"friend": FRIEND, "quick_test": QUICK_TEST, "results": rows, "expected": EXPECT},
          open("/kaggle/working/runs/sweep_result.json", "w"), indent=1, default=str)
for r, out, p in procs:
    if os.path.exists(f"{out}/curve.png"): display(Image(f"{out}/curve.png"))
import shutil
shutil.rmtree("/kaggle/working/vec2vec", ignore_errors=True)   # do not keep the code copy in the output
for r, out, p in procs:
    print(r["name"], sorted(os.listdir(out)))
if QUICK_TEST:
    print("\nQUICK TEST PASSED" if all(p.returncode == 0 for _, _, p in procs) else "\nQUICK TEST FAILED - send Aman the error above")
    print("Now set QUICK_TEST = False and do Save Version -> Save & Run All (Commit).")
""")

md("""
## Reading the result (for Aman)

- `LEARNED` uses the rule written in cell 3 before the run.
- `top1_final` is the honest number (what an attacker gets without the answers). `top1_max` / `avg_rank_min`
  look at every epoch with the true pairs, so they are only used to decide whether the GAN started learning.
- `resume.pt` lets the workstation continue exactly this run (same seed, same optimiser state) for days.
""")

nb = nbf.v4.new_notebook()
nb["cells"] = cells
nb["metadata"] = {"kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"},
                  "language_info": {"name": "python"}}
nbf.write(nb, HERE / "phaseV2_vec2vec_sweep.ipynb")
print("written", len(cells), "cells")
