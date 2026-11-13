import nbformat as nbf
from pathlib import Path

HERE = Path(__file__).parent
cells = []
md = lambda s: cells.append(nbf.v4.new_markdown_cell(s.strip()))
code = lambda s: cells.append(nbf.v4.new_code_cell(s.strip()))

md("""
# Phase C (v2, same-source attacker): mini-vec2vec attack on each compression scheme

Neural Nexus, vector-DB compression project (team leader: Aman).

| | |
|---|---|
| **Time** | about 5 min setup + 5 min control, then about 6-10 min per scheme per seed (5 seeds): about 1 h per scheme |
| **Settings** | Accelerator = **None (CPU)**, Internet = **On** |
| **Inputs** | Add Input -> Notebooks: `phaseA_fp32_embeddings`, and the Phase B notebook of every scheme you run (not needed for fp32). A2b is NOT needed any more |
| **The only line you change** | `SCHEMES = [...]` in cell 2 |
| **How to run** | Save Version -> Save & Run All (Commit), close the tab |
| **Output** | `phaseC/`: results_C_<schemes>.json, per-run W matrices, translated victim vectors for Phase D, plots |

What it does:
1. **Control first**: runs mini-vec2vec on the paper's own data (e5 -> gtr, 60k NQ sentences). The paper reports
   top-1 0.96. If our code gets less than 0.85 there, the notebook stops: then a bad number later would mean a bug, not a result.
2. For each scheme: the attacker has the stolen victim vectors (BGE-large, decoded from the compressed database) and
   19,569 of their own GTR vectors from the SAME text source as the victim (Phase A attacker passages, no
   overlap). No pairs. mini-vec2vec learns a linear map BGE -> GTR. 5 seeds.
   Why v2: v1 used the 200k DPR set and failed on every scheme (fp32 top-1 0.0002), while the same-source set
   gave 0.55 / 0.91 / 0.95. The attacker's text source matters more than its size.
3. Scores on the fixed 8,192 victim passages against their true GTR vectors (same metric as every phase).
4. Leak check first: no attacker passage may be a (near-)copy of a victim passage (it would inflate the result).
   (The attacker-set check and supervised bounds already ran in v1 and are not repeated.)
""")

code(r"""
# Cell 1 - settings. CHANGE ONLY SCHEMES.
SCHEMES = ["fp32"]          # any of: "fp32", "int8", "int4", "pq128x8", "binary", "svd256"
SEEDS = [0, 1, 2, 3, 4]
L, MODEL, EVAL_N = 32, "bge-large", 8192

import os, glob, json, time, sys
import numpy as np, torch, scipy, sklearn
torch.set_num_threads(os.cpu_count())
OUT = "/kaggle/working/phaseC2"; os.makedirs(OUT, exist_ok=True)
print("schemes", SCHEMES, "| cpus", os.cpu_count(), "| scipy", scipy.__version__, "| sklearn", sklearn.__version__, "| torch", torch.__version__)
""")

code(r"""
# Cell 2 - what we expect BEFORE the run (written down so results cannot move the goalposts)
EXPECT = {
    "random guess": "top-1 = 1/8,192 = 0.00012, average rank = 4,096",
    "control e5->gtr (paper Table 1)": "top-1 0.96, average rank 1.10, cosine 0.22 / 0.49 / 0.51 (initial / refine-1 / refine-2). "
                                       "GATE: ours must reach top-1 >= 0.85, else the notebook stops",
    "fp32 BGE-large -> GTR (same-source attacker)": "v1 check already gave top-1 0.55 / 0.91 / 0.95 (seeds 0-2). "
                             "Expect mean about 0.8 over 5 seeds, with 0-2 weak seeds",
    "leak check": "victim passages with a centred GTR cosine > 0.95 to any attacker passage: expect under 0.5% (A2b found 0.08% for DPR)",
    "self-consistency": "should be higher for good seeds than for weak ones (Spearman > 0.5 across all runs)",
    "int8, int4 vs fp32": "within 0.02 top-1 (decoded cosine 0.99997 / 0.992: almost the same vectors)",
    "pq128x8 vs fp32": "lower; decoded cosine 0.90. Guess: 0.0-0.2 below fp32",
    "binary vs fp32": "lower again; decoded cosine 0.79. Guess: 0.1-0.4 below fp32",
    "svd256 vs fp32": "unknown: 256 -> 768 map, 89% of the energy kept",
    "spread over seeds": "larger than the paper's 0.00-0.01 (v1 check: 0.55-0.95)",
    "supervised upper bound (v1, with pairs)": "Procrustes 0.993, ridge 0.994, MLP 0.993 top-1",
}
for k, v in EXPECT.items(): print(f"- {k}: {v}")
""")

code((HERE / "minivec2vec.py").read_text().replace("import numpy as np, torch", "import numpy as np, torch  # noqa", 1)
     + "\nprint('mini-vec2vec code loaded')")

code(r"""
# Cell 4 - CONTROL: the paper's own data (dar-tau/nq-embeddings), official split. About 5-10 minutes.
from huggingface_hub import hf_hub_download
t0 = time.time()
emb = {k: torch.load(hf_hub_download("dar-tau/nq-embeddings", f"{k}.pt"), map_location="cpu").float().numpy()
       for k in ["e5", "gtr"]}
A, B = emb["e5"], emb["gtr"]
n = len(A) - 8192
mA, mB = A[:n].mean(0), B[:n].mean(0)
XA_c, XB_c = prep(A[:n // 2], mA), prep(B[n // 2:n], mB)      # different sentences on the two sides (official)
EA_c, EB_c = prep(A[n:], mA), prep(B[n:], mB)
print("control sizes:", XA_c.shape, XB_c.shape, EA_c.shape)
W_c, hist_c = mini_vec2vec(XA_c, XB_c, seed=0, eval_fn=lambda W: align_eval(EA_c, EB_c, W))
control = {"hist": hist_c, "minutes": (time.time() - t0) / 60, "paper_top1": 0.96}
json.dump(control, open(f"{OUT}/control_e5_gtr.json", "w"), indent=1)
print(json.dumps(control, indent=1))
assert hist_c["refine2"]["top1"] >= 0.85, "CONTROL FAILED: our mini-vec2vec does not reproduce the paper. Stop and tell Aman."
del A, B, emb, XA_c, XB_c
print("CONTROL PASSED")
""")

code(r"""
# Cell 5 - load our data and check it
def find_one(pattern, what):
    hits = sorted(glob.glob(f"/kaggle/input/**/{pattern}", recursive=True))
    assert hits, f"{what}: '{pattern}' not found. Add the input notebook (see the table at the top)"
    if len(hits) > 1:
        assert len({os.path.getsize(h) for h in hits}) == 1, f"{pattern}: different copies found: {hits}"
    return hits[0]

IN_A = os.path.dirname(find_one("manifest.json", "Phase A"))
assert json.load(open(f"{IN_A}/manifest.json"))["phase"] == "A_fp32_embeddings"
Y_true = np.load(find_one(f"gtr_victim_L{L}.npy", "Phase A")).astype(np.float32)        # scoring only
Y_att = np.load(find_one(f"gtr_att_L{L}.npy", "Phase A")).astype(np.float32)     # attacker's own vectors (same source)
eval_idx = np.random.default_rng(0).choice(50000, EVAL_N, replace=False)               # same 8,192 as every phase
assert Y_true.shape == (50000, 768) and Y_att.shape[1] == 768 and len(Y_att) > 15000
# leak check: is any attacker passage a near-copy of a victim passage? GTR cosine after removing the common
# direction (centred + unit length; raw GTR vectors all point roughly the same way, which would hide copies)
_m = Y_att.mean(0)
_a, _v = torch.from_numpy(prep(Y_att, _m)), torch.from_numpy(prep(Y_true, _m))
_mx = torch.cat([(_v[i:i + 4096] @ _a.T).max(1).values for i in range(0, len(_v), 4096)]).numpy()
leak = {"victim_with_attacker_cos_gt_0.95": float((_mx > 0.95).mean()), "victim_with_attacker_cos_gt_0.90": float((_mx > 0.90).mean()),
        "median_nearest_cos": float(np.median(_mx))}
print("leak check:", leak)
assert leak["victim_with_attacker_cos_gt_0.95"] < 0.005, "attacker set overlaps the victim: stop and tell Aman"
del _a, _v
mean_B = Y_att.mean(0)
XB = prep(Y_att, mean_B)
norm_B = float(np.median(np.linalg.norm(Y_att - mean_B, axis=1)))   # to map translations back to raw GTR scale (Phase D)
EB = prep(Y_true[eval_idx], mean_B)

def victim_vectors(scheme):
    f = f"{MODEL}_victim_L{L}.npy" if scheme == "fp32" else f"decoded_victim_{scheme}_{MODEL}_L{L}.npy"
    X = np.load(find_one(f, f"Phase B {scheme}")).astype(np.float32)
    assert len(X) == 50000 and np.isfinite(X).all(), (scheme, X.shape)
    return X

for s in SCHEMES:
    print(s, victim_vectors(s).shape)
print("attacker (same source)", Y_att.shape, "| eval", EB.shape)
""")

code(r"""
# Cell 6 - the attack: every scheme x every seed. This is the long cell.
results = {"version": "v2_same_source_attacker", "control": control, "leak_check": leak, "runs": [], "expected": EXPECT,
           "norm_B": norm_B, "n_attacker": len(Y_att)}
np.save(f"{OUT}/mean_B.npy", mean_B)
for scheme in SCHEMES:
    Xv = victim_vectors(scheme)
    mean_A = Xv.mean(0)                    # the attacker can compute this from the stolen vectors
    XA, EA = prep(Xv, mean_A), prep(Xv[eval_idx], mean_A)
    np.save(f"{OUT}/mean_A_{scheme}.npy", mean_A)
    for seed in SEEDS:
        t0 = time.time()
        print(f"\n=== {scheme} seed {seed} ===")
        W, hist = mini_vec2vec(XA, XB, seed=seed, eval_fn=lambda W: align_eval(EA, EB, W))
        np.save(f"{OUT}/W_{scheme}_s{seed}.npy", W)
        run = {"scheme": scheme, "seed": seed, "dim_victim": XA.shape[1], "minutes": (time.time() - t0) / 60, **hist}
        results["runs"].append(run)
        print(f"{scheme} seed {seed}: top-1 {hist['refine2']['top1']:.4f}  rank {hist['refine2']['avg_rank']:.1f}  "
              f"cos {hist['refine2']['mean_cos']:.3f}  self-consistency {hist['self_consistency']:.3f}  ({run['minutes']:.1f} min)")
        json.dump(results, open(f"{OUT}/results_C_{'_'.join(SCHEMES)}.json", "w"), indent=1)
    # Phase D input: the translated 8,192 eval vectors in raw GTR scale, from the seed an attacker would pick
    # WITHOUT the answers (highest self-consistency), so Phase D measures an honest attack.
    runs = [r for r in results["runs"] if r["scheme"] == scheme]
    pick = max(runs, key=lambda r: r["self_consistency"])
    W = np.load(f"{OUT}/W_{scheme}_s{pick['seed']}.npy")
    Y_hat = l2n(EA @ W) * norm_B + mean_B
    np.save(f"{OUT}/translated_eval_{scheme}.npy", Y_hat.astype(np.float32))
    results.setdefault("phaseD_pick", {})[scheme] = {"seed": pick["seed"], "picked_by": "highest self_consistency",
                                                     "top1": pick["refine2"]["top1"]}
json.dump(results, open(f"{OUT}/results_C_{'_'.join(SCHEMES)}.json", "w"), indent=1)
np.save(f"{OUT}/eval_idx.npy", eval_idx)
""")

code(r"""
# Cell 7 - fp32 only: (a) attacker-set check, (b) supervised upper bounds (attacker WITH pairs)
if False:   # v1 extras (attacker-set check, supervised bounds) already done; not repeated
    Xv = victim_vectors("fp32"); mean_A = Xv.mean(0)
    XA, EA = prep(Xv, mean_A), prep(Xv[eval_idx], mean_A)
    # (a) same-source attacker set from Phase A (same text source as the victim, smaller)
    Y_small = np.load(find_one(f"gtr_att_L{L}.npy", "Phase A")).astype(np.float32)
    mB_s = Y_small.mean(0); XB_s, EB_s = prep(Y_small, mB_s), prep(Y_true[eval_idx], mB_s)
    small = []
    for seed in SEEDS:
        W, hist = mini_vec2vec(XA, XB_s, seed=seed, eval_fn=lambda W: align_eval(EA, EB_s, W), log=lambda *a: None)
        small.append(hist["refine2"]); print("same-source set, seed", seed, hist["refine2"])
    big = [r["refine2"]["top1"] for r in results["runs"] if r["scheme"] == "fp32"]
    gap = float(np.mean([s["top1"] for s in small]) - np.mean(big))
    results["attacker_set_check"] = {"n_same_source": len(Y_small), "same_source": small, "top1_200k": big,
                                     "same_source_minus_200k": gap,
                                     "rule": "use 200k only if within 0.02", "use_200k": bool(gap <= 0.02)}
    print(json.dumps(results["attacker_set_check"], indent=1))

    # (b) supervised: the attacker's OWN passages embedded with BOTH models (pairs). Upper bound only.
    Xp = prep(np.load(find_one(f"{MODEL}_att_L{L}.npy", "Phase A")), mean_A)
    Yp = prep(Y_small, mean_B)
    assert len(Xp) == len(Yp), "attacker pairs must line up"
    Ep = prep(Y_true[eval_idx], mean_B)
    sup = {"n_pairs": len(Xp)}
    sup["procrustes"] = align_eval(EA, Ep, procrustes(Xp, Yp))
    lam = 1e-2   # ridge regression, closed form
    Wr = np.linalg.solve(Xp.T @ Xp + lam * np.eye(Xp.shape[1]), Xp.T @ Yp).astype(np.float32)
    sup["ridge"] = align_eval(EA, Ep, Wr)
    torch.manual_seed(0)
    mlp = torch.nn.Sequential(torch.nn.Linear(1024, 2048), torch.nn.GELU(), torch.nn.Linear(2048, 768))
    opt = torch.optim.AdamW(mlp.parameters(), lr=1e-3, weight_decay=1e-4)
    Xt, Yt = torch.from_numpy(Xp), torch.from_numpy(Yp)
    for epoch in range(30):
        perm = torch.randperm(len(Xt))
        for i in range(0, len(Xt), 256):
            b = perm[i:i + 256]
            loss = 1 - torch.nn.functional.cosine_similarity(mlp(Xt[b]), Yt[b]).mean()
            opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        Z = mlp(torch.from_numpy(EA)).numpy()
    sup["mlp"] = align_eval(Z, Ep, np.eye(768, dtype=np.float32))
    results["supervised"] = sup
    print(json.dumps(sup, indent=1))
    json.dump(results, open(f"{OUT}/results_C_{'_'.join(SCHEMES)}.json", "w"), indent=1)
""")

code(r"""
# Cell 8 - summary table and plots
import pandas as pd, matplotlib.pyplot as plt
rows = [{"scheme": r["scheme"], "seed": r["seed"], **{f"{st}_top1": r[st]["top1"] for st in ["initial", "refine1", "refine2"]},
         "avg_rank": r["refine2"]["avg_rank"], "mean_cos": r["refine2"]["mean_cos"],
         "self_consistency": r["self_consistency"], "minutes": r["minutes"]} for r in results["runs"]]
df = pd.DataFrame(rows); print(df.round(4).to_string(index=False))
from scipy.stats import spearmanr
sc = spearmanr(df["self_consistency"], df["refine2_top1"])
results["self_consistency_vs_top1_spearman"] = float(sc.correlation)
print("self-consistency vs top-1, Spearman:", round(sc.correlation, 3), "| per-scheme pick for Phase D:", results["phaseD_pick"])
agg = df.groupby("scheme", sort=False).agg(top1_mean=("refine2_top1", "mean"), top1_std=("refine2_top1", "std"),
                                           rank_mean=("avg_rank", "mean"), cos_mean=("mean_cos", "mean"))
print(agg.round(4)); results["summary"] = agg.reset_index().to_dict("records")
json.dump(results, open(f"{OUT}/results_C_{'_'.join(SCHEMES)}.json", "w"), indent=1)

fig, ax = plt.subplots(1, 2, figsize=(13, 4.5))
ax[0].bar(agg.index, agg["top1_mean"], yerr=agg["top1_std"].fillna(0), color="tab:red", capsize=4,
          label="mini-vec2vec (mean of 3 seeds)")
ax[0].axhline(1 / EVAL_N, color="gray", ls="--", lw=1, label="random guess")
ax[0].axhline(0.993, color="tab:green", ls=":", label="supervised linear with pairs (v1, fp32)")
ax[0].set_xlabel("Compression scheme of the stolen database"); ax[0].set_ylabel("Top-1 accuracy (8,192 candidates)")
ax[0].set_title("mini-vec2vec translation accuracy per scheme"); ax[0].legend(); ax[0].set_ylim(0, 1.05)
for scheme, g in df.groupby("scheme", sort=False):
    ax[1].plot(["initial", "refine-1", "refine-2"], g[["initial_top1", "refine1_top1", "refine2_top1"]].mean().values,
               marker="o", label=scheme)
ax[1].plot(["initial", "refine-1", "refine-2"], [control["hist"][s]["top1"] for s in ["initial", "refine1", "refine2"]],
           marker="s", ls="--", color="black", label="control e5->gtr (paper data)")
ax[1].set_xlabel("mini-vec2vec stage"); ax[1].set_ylabel("Top-1 accuracy"); ax[1].set_ylim(0, 1.05)
ax[1].set_title("Top-1 after each stage (mean of seeds)"); ax[1].legend(); ax[1].grid(alpha=0.3)
plt.tight_layout(); plt.savefig(f"{OUT}/phaseC_{'_'.join(SCHEMES)}.png", dpi=140); plt.show()
print(sorted(os.listdir(OUT)))
""")

md("""
## Reading the result (for Aman)
- Control passed means the code reproduces the paper; every number below is then a real result, not a bug.
- `refine2_top1` is the honest attack result. Seeds show how stable it is.
- `phaseD_pick` is the seed Phase D will invert. It was chosen by self-consistency, which needs no answers.
""")

nb = nbf.v4.new_notebook()
nb["cells"] = cells
nb["metadata"] = {"kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"},
                  "language_info": {"name": "python"}}
nbf.write(nb, HERE / "phaseC2_minivec2vec_samesource.ipynb")
print("written", len(cells), "cells")
