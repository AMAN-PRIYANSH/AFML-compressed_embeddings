import nbformat as nbf
from pathlib import Path

HERE = Path(__file__).parent
cells = []
md = lambda s: cells.append(nbf.v4.new_markdown_cell(s.strip()))
code = lambda s: cells.append(nbf.v4.new_code_cell(s.strip()))

md("""
# Phase V: the original vec2vec on our data, with a fixed time budget

Neural Nexus, vector-DB compression project. Two runs in total: `SCHEME = "fp32"` and `SCHEME = "pq128x8"`.

Why: our main attacker is mini-vec2vec (linear, minutes on a CPU). This notebook measures how long the
original vec2vec (a GAN, Jha et al. 2025) needs on OUR data, and how good it gets, so the paper can show
the comparison with real numbers instead of quoting "1 to 7 days".

What it does:
1. Downloads the authors' official vec2vec code (fixed commit 5efb3c0) and runs THEIR trainer with THEIR
   settings (configs/unsupervised.toml). Only the data input is changed: it reads our saved vectors.
2. Victim side: the 50,000 victim BGE-large vectors (32 tokens), uncompressed or PQ-decoded.
   Attacker side: the 200,000 A2 GTR vectors. No pairs are given to training.
3. After every epoch it scores the translation on 8,192 victim passages against their true GTR vectors
   (top-1, average rank, mean cosine). Same metric as mini-vec2vec, so the two are directly comparable.
4. Stops after 10.5 hours, or earlier if the authors' own early-stopping rule fires.
5. Saves the log, the latest and the best translator, a plot, and a summary.

Settings: Accelerator = GPU T4 (x1 is enough), Internet = On.
Inputs (Add Input -> Notebooks): `phaseA_fp32_embeddings`, `phaseA2b_verify_attacker_extra`,
and for `pq128x8` also the teammate's `phaseB-pq128x8` notebook.
Run with Save Version -> Save & Run All (Commit). Expected time: up to about 11 hours.
""")

code(r"""
# Cell 1 - install + official vec2vec code at a fixed commit
!pip install -q "transformers==4.46.3" "vec2text==0.0.13" "sentence-transformers==3.4.1" accelerate toml wandb evaluate diffusers nltk seaborn
!rm -rf /kaggle/working/vec2vec && git clone -q https://github.com/rjha18/vec2vec.git /kaggle/working/vec2vec && cd /kaggle/working/vec2vec && git checkout -q 5efb3c02ee908847581c6488cf9c8ed6dace3c29 && git log -1 --format='vec2vec commit %H %cd'
""")

code(r"""
# Cell 2 - settings. CHANGE ONLY SCHEME.
SCHEME = "fp32"          # "fp32" or "pq128x8"
BUDGET_HOURS = 10.5      # Kaggle stops a session at 12 h; keep ~1 h for setup and saving
SEED = 5                 # the authors' default seed in configs/unsupervised.toml
EVAL_N = 8192            # same evaluation size as the vec2vec and mini-vec2vec papers
L, MODEL = 32, "bge-large"

import os, glob, json, time
import numpy as np, torch
assert torch.cuda.is_available(), "Set Accelerator = GPU T4"
OUT = f"/kaggle/working/phaseV_{SCHEME}"
os.makedirs(OUT, exist_ok=True)

def find_one(pattern, what):
    hits = glob.glob(f"/kaggle/input/**/{pattern}", recursive=True)
    assert len(hits) == 1, f"{what}: expected exactly one '{pattern}' in the inputs, found {hits}"
    return hits[0]
IN_A = os.path.dirname(find_one("manifest.json", "Phase A output"))
IN_A2 = os.path.dirname(find_one("manifest_A2.json", "Phase A2b output"))
assert json.load(open(f"{IN_A}/manifest.json"))["phase"] == "A_fp32_embeddings"
if SCHEME == "fp32":
    victim_path = f"{IN_A}/{MODEL}_victim_L{L}.npy"
else:
    victim_path = find_one(f"decoded_victim_{SCHEME}_{MODEL}_L{L}.npy", f"Phase B {SCHEME} output")
print("victim vectors:", victim_path)
""")

code(r"""
# Cell 3 - what we expect BEFORE the run (so the run cannot move the goalposts)
EXPECT = {
    "setup (install + clone + load)": "5-10 minutes",
    "steps per epoch": "50,000 / 256 = 195 (the smaller side sets the epoch length)",
    "minutes per epoch on a T4": "unknown before this run; rough guess 0.5-2",
    "random-guess top-1 (8,192 candidates)": "about 0.0001",
    "if training works (fp32)": "top-1 rises clearly above 0.01 within the first 1-2 hours and ends around 0.5-0.85. "
                                "The paper reports 0.82-0.99 with 1M vectors per side; its own data ablation gives ~0.81 with 50k on one side",
    "if the GAN fails for this seed": "top-1 stays near 0.0001 for hours. The paper itself says cross-backbone runs often fail "
                                      "(e5->gtr: 3 of 15 seeds converged in 10 epochs). That is a result, not a bug",
    "pq128x8 vs fp32": "equal or lower top-1 than fp32",
    "stop reason": "time budget (10.5 h) or the authors' early stopping (after epoch 80, no improvement for 20 epochs)",
}
for k, v in EXPECT.items(): print(f"- {k}: {v}")
""")

code(r"""
# Cell 4 - load data and check it
X_victim = np.load(victim_path).astype(np.float32)
Y_att = np.load(f"{IN_A2}/gtr_attextra_L{L}.npy").astype(np.float32)
Y_true = np.load(f"{IN_A}/gtr_victim_L{L}.npy").astype(np.float32)    # scoring only, never trained on
eval_idx = np.random.default_rng(0).choice(len(X_victim), EVAL_N, replace=False)
X_eval, Y_eval = X_victim[eval_idx], Y_true[eval_idx]
print("victim", X_victim.shape, "attacker", Y_att.shape, "eval", X_eval.shape, Y_eval.shape)
assert X_victim.shape == (50000, 1024) and Y_att.shape == (200000, 768) and Y_true.shape == (50000, 768)
assert np.isfinite(X_victim).all() and np.isfinite(Y_att).all()
if SCHEME == "fp32":
    assert abs(np.linalg.norm(X_victim, axis=1).mean() - 1) < 1e-3
""")

code((HERE / "v2v_runner.py").read_text())

code(r"""
# Cell 6 - run (this is the long cell)
t0 = time.time()
best, log_path = run("/kaggle/working/vec2vec", X_victim, Y_att, X_eval, Y_eval, OUT,
                     budget_hours=BUDGET_HOURS, seed=SEED)
print(f"finished after {(time.time()-t0)/3600:.2f} h; best: {best}")
""")

code(r"""
# Cell 7 - results: table, plot, summary
import pandas as pd, matplotlib.pyplot as plt
os.chdir("/kaggle/working")
rows = [json.loads(l) for l in open(log_path)]
ours = pd.DataFrame([r for r in rows if r["kind"] == "ours"])
last = ours.iloc[-1].to_dict()
summary = {"scheme": SCHEME, "seed": SEED, "budget_hours": BUDGET_HOURS,
           "hours_used": float(ours["t_hours"].iloc[-1]), "epochs": int(ours["epoch"].iloc[-1]),
           "minutes_per_epoch_median": float(ours["epoch_minutes"].median()),
           "final": {k: float(last[k]) for k in ["top1", "avg_rank", "mean_cos"]},
           "best_by_true_pairs": best,
           "note": "final = honest unsupervised result; best = picked using the true pairs (optimistic upper bound)",
           "expected": EXPECT}
json.dump(summary, open(f"{OUT}/summary_{SCHEME}.json", "w"), indent=1)
print(json.dumps(summary, indent=1))

fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
ax[0].plot(ours["t_hours"], ours["top1"], color="tab:red", label="vec2vec top-1")
ax[0].axhline(1 / EVAL_N, color="gray", ls="--", lw=1, label="random guess")
ax[0].set_xlabel("Training time (hours, T4)"); ax[0].set_ylabel("Top-1 accuracy (8,192 candidates)")
ax[0].set_title(f"vec2vec on {SCHEME}: top-1 vs time"); ax[0].legend(); ax[0].grid(alpha=0.3)
ax[1].plot(ours["t_hours"], ours["mean_cos"], color="tab:blue", label="vec2vec mean cosine")
ax[1].set_xlabel("Training time (hours, T4)"); ax[1].set_ylabel("Mean cosine to true GTR vector")
ax[1].set_title(f"vec2vec on {SCHEME}: cosine vs time"); ax[1].legend(); ax[1].grid(alpha=0.3)
plt.tight_layout(); plt.savefig(f"{OUT}/curve_{SCHEME}.png", dpi=150); plt.show()
shutil.rmtree("/kaggle/working/vec2vec", ignore_errors=True)   # do not keep the code copy in the output
print(sorted(os.listdir(OUT)))
""")

md("""
## Reading the result

- `final.top1` is the honest number: what an attacker gets after this much training, with no access to pairs.
- `best_by_true_pairs` picks the best epoch using the real answers. A real attacker cannot do that, so it is an upper bound.
- `hours_used` and `epochs` are how long vec2vec needed (or got) on our data. Compare with mini-vec2vec in Phase C.
- If top-1 stayed near 0.0001 the whole time, the GAN did not converge for this seed. Report it as is; do not rerun silently.
""")

nb = nbf.v4.new_notebook()
nb["cells"] = cells
nb["metadata"] = {"kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"},
                  "language_info": {"name": "python"}}
nbf.write(nb, HERE / "phaseV_vec2vec_budget.ipynb")
print("written", len(cells), "cells")
