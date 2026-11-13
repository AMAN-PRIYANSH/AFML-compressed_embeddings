"""Phase V on a local machine (workstation). Same experiment as phaseV_vec2vec_budget.ipynb on Kaggle:
same official vec2vec code and settings, same data files, same seed, same 8,192 evaluation passages.
Only the machine, the time budget and the folder paths differ, and the summary records the GPU used.

Usage (from the vec2vec-run folder):
  uv run python run_v2v_local.py --scheme fp32 --hours 36
  uv run python run_v2v_local.py --scheme pq128x8 --hours 36
  uv run python run_v2v_local.py --scheme fp32 --hours 0.1 --smoke     (5-10 minute test on a small subset)
"""
import os, sys, json, time, argparse, platform

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

p = argparse.ArgumentParser()
p.add_argument("--scheme", required=True, choices=["fp32", "pq128x8"])
p.add_argument("--hours", type=float, required=True, help="training time budget")
p.add_argument("--data", default=os.path.join(HERE, "data"), help="folder with the downloaded .npy files")
p.add_argument("--out", default=None, help="output folder (default: outputs/<scheme>)")
p.add_argument("--repo", default=os.path.join(HERE, "vec2vec"), help="clone of github.com/rjha18/vec2vec")
p.add_argument("--seed", type=int, default=5)
p.add_argument("--smoke", action="store_true", help="tiny subset, to test the setup")
a = p.parse_args()

import numpy as np, torch
from v2v_runner import run

OUT = os.path.abspath(a.out or os.path.join(HERE, "outputs", a.scheme + ("_smoke" if a.smoke else "")))
os.makedirs(OUT, exist_ok=True)

# keep Windows awake while this process runs (no admin rights needed; the screen may still turn off)
if os.name == "nt":
    import ctypes
    ctypes.windll.kernel32.SetThreadExecutionState(0x80000000 | 0x00000001)  # ES_CONTINUOUS | ES_SYSTEM_REQUIRED

assert torch.cuda.is_available() or a.smoke, "No CUDA GPU visible to PyTorch"
L, MODEL, EVAL_N = 32, "bge-large", 8192
files = {
    "victim": f"{MODEL}_victim_L{L}.npy" if a.scheme == "fp32" else f"decoded_victim_{a.scheme}_{MODEL}_L{L}.npy",
    "attacker": f"gtr_attextra_L{L}.npy",
    "true_gtr": f"gtr_victim_L{L}.npy",
}
for k, f in files.items():
    assert os.path.exists(os.path.join(a.data, f)), f"missing {f} in {a.data} (see README step 4)"

X_victim = np.load(os.path.join(a.data, files["victim"])).astype(np.float32)
Y_att = np.load(os.path.join(a.data, files["attacker"])).astype(np.float32)
Y_true = np.load(os.path.join(a.data, files["true_gtr"])).astype(np.float32)
assert X_victim.shape == (50000, 1024) and Y_att.shape == (200000, 768) and Y_true.shape == (50000, 768)
assert np.isfinite(X_victim).all() and np.isfinite(Y_att).all()
eval_idx = np.random.default_rng(0).choice(len(X_victim), EVAL_N, replace=False)   # same as Kaggle notebook
X_eval, Y_eval = X_victim[eval_idx], Y_true[eval_idx]
if a.smoke:
    X_victim, Y_att, X_eval, Y_eval = X_victim[:5000], Y_att[:20000], X_eval[:1024], Y_eval[:1024]

gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
print(f"scheme={a.scheme} gpu={gpu} budget={a.hours}h out={OUT}", flush=True)
extra = {} if torch.cuda.is_available() else {"mixed_precision": "no"}
if a.smoke:
    extra.update({"val_bs": 512, "top_k_size": 512, "top_k_batches": 2})

t0 = time.time()
best, log_path = run(a.repo, X_victim, Y_att, X_eval, Y_eval, OUT, budget_hours=a.hours, seed=a.seed,
                     val_size=min(4096, len(X_eval)), extra_args=extra, log=lambda s: print(s, flush=True))
os.chdir(HERE)

rows = [json.loads(l) for l in open(log_path)]
ours = [r for r in rows if r["kind"] == "ours"]
summary = {"scheme": a.scheme, "seed": a.seed, "budget_hours": a.hours, "smoke_test": a.smoke,
           "machine": platform.node(), "gpu": gpu, "torch": torch.__version__,
           "hours_used": ours[-1]["t_hours"], "epochs": ours[-1]["epoch"],
           "minutes_per_epoch_median": float(np.median([r["epoch_minutes"] for r in ours])),
           "final": {k: ours[-1][k] for k in ["top1", "avg_rank", "mean_cos"]},
           "best_by_true_pairs": best,
           "note": "final = honest unsupervised result; best = picked using the true pairs (optimistic upper bound)"}
json.dump(summary, open(os.path.join(OUT, f"summary_{a.scheme}.json"), "w"), indent=1)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
t = [r["t_hours"] for r in ours]
ax[0].plot(t, [r["top1"] for r in ours], color="tab:red", label="vec2vec top-1")
ax[0].axhline(1 / len(X_eval), color="gray", ls="--", lw=1, label="random guess")
ax[0].set_xlabel(f"Training time (hours, {gpu})"); ax[0].set_ylabel(f"Top-1 accuracy ({len(X_eval):,} candidates)")
ax[0].set_title(f"vec2vec on {a.scheme}: top-1 vs time"); ax[0].legend(); ax[0].grid(alpha=0.3)
ax[1].plot(t, [r["mean_cos"] for r in ours], color="tab:blue", label="vec2vec mean cosine")
ax[1].set_xlabel(f"Training time (hours, {gpu})"); ax[1].set_ylabel("Mean cosine to true GTR vector")
ax[1].set_title(f"vec2vec on {a.scheme}: cosine vs time"); ax[1].legend(); ax[1].grid(alpha=0.3)
plt.tight_layout(); plt.savefig(os.path.join(OUT, f"curve_{a.scheme}.png"), dpi=150)
print("DONE", json.dumps(summary), flush=True)
