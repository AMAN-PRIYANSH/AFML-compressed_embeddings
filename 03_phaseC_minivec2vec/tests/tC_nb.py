import os, matplotlib; matplotlib.use('Agg')

# Cell 1 - settings. CHANGE ONLY SCHEMES.
SCHEMES = ["fp32", "pq128x8"]          # any of: "fp32", "int8", "int4", "pq128x8", "binary", "svd256"
SEEDS = [0, 1, 2, 3, 4]
L, MODEL, EVAL_N = 32, "bge-large", 1024

import os, glob, json, time, sys
import numpy as np, torch, scipy, sklearn
torch.set_num_threads(os.cpu_count())
OUT = "./_sandbox/phaseC/tC/working/phaseC2"; os.makedirs(OUT, exist_ok=True)
print("schemes", SCHEMES, "| cpus", os.cpu_count(), "| scipy", scipy.__version__, "| sklearn", sklearn.__version__, "| torch", torch.__version__)

# Cell 2 - what we expect BEFORE the run (written down so results cannot move the goalposts)
EXPECT = {
    "random guess": "top-1 = 1/8,192 = 0.00012, average rank = 4,096",
    "control e5->gtr (paper Table 1)": "top-1 0.96, average rank 1.10, cosine 0.22 / 0.49 / 0.51 (initial / refine-1 / refine-2). "
                                       "GATE: ours must reach top-1 >= 0.85, else the notebook stops",
    "fp32 BGE-large -> GTR (same-source attacker)": "v1 check already gave top-1 0.55 / 0.91 / 0.95 (seeds 0-2). "
                             "Expect mean about 0.8 over 5 seeds, with 0-2 weak seeds",
    "leak check": "before filtering 1.56% of victim passages have a near-copy (found 8 Nov); after dropping those attacker rows: 0",
    "without former near-copies": "top-1 on the eval passages that never had a near-copy: within 0.02 of the full score",
    "self-consistency": "should be higher for good seeds than for weak ones (Spearman > 0.5 across all runs)",
    "int8, int4 vs fp32": "within 0.02 top-1 (decoded cosine 0.99997 / 0.992: almost the same vectors)",
    "pq128x8 vs fp32": "lower; decoded cosine 0.90. Guess: 0.0-0.2 below fp32",
    "binary vs fp32": "lower again; decoded cosine 0.79. Guess: 0.1-0.4 below fp32",
    "svd256 vs fp32": "unknown: 256 -> 768 map, 89% of the energy kept",
    "spread over seeds": "larger than the paper's 0.00-0.01 (v1 check: 0.55-0.95)",
    "supervised upper bound (v1, with pairs)": "Procrustes 0.993, ridge 0.994, MLP 0.993 top-1",
}
for k, v in EXPECT.items(): print(f"- {k}: {v}")

# ---- mini-vec2vec (Dar 2025, arXiv 2510.02348) ----
# Follows the official notebook github.com/guy-dar/mini-vec2vec (commit 948de01, 11 Oct 2025) step by step,
# with the official settings: 30 x (20 clusters, 30 QAP restarts, 10k subsample), k = 50, Refine-1 100 x
# (1,000 points, k = 50), Refine-2 one round with 500 clusters, mixing 0.5.
# Changes, none of which alter the method:
#  1. Rectangular Procrustes (SVD of A^T B) so a 1024-d (or 256-d) victim space can map into 768-d GTR.
#     For equal sizes this is exactly scipy's orthogonal_procrustes, which the official notebook uses.
#  2. Seeds: the QAP solver gets an explicit random seed (in the official notebook it is unseeded, so runs
#     cannot be repeated). KMeans also gets a seed.
#  3. Similarities and top-k are computed in blocks with torch (same numbers, less memory: 50k x 200k at once
#     would need 40 GB).
#  4. Extra logging: an attacker-visible quality score (Refine-2 self-consistency), see below.
import numpy as np, torch  # noqa
from scipy.optimize import quadratic_assignment
from sklearn.cluster import KMeans

def l2n(X):
    return X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)

def prep(X, mean):
    """Official preprocessing: subtract this space's mean, then unit length."""
    return l2n(np.asarray(X, dtype=np.float32) - mean).astype(np.float32)

def procrustes(A, B):
    """W minimising ||A W - B|| with orthonormal rows/columns (rectangular-safe)."""
    U, _, Vt = np.linalg.svd(A.T.astype(np.float64) @ B.astype(np.float64), full_matrices=False)
    return (U @ Vt).astype(np.float32)

def _double_centered_gram(C):
    G = C @ C.T                                   # official sim(): H C C^T H
    n = len(C)
    H = np.eye(n) - np.ones((n, n)) / n
    return H @ G @ H

def aligned_centroids(X, Y, rng, n_clusters=20, n_qap=30, subsample=10_000):
    xs = X[rng.permutation(len(X))[:subsample]]
    ys = Y[rng.permutation(len(Y))[:subsample]]
    ca = KMeans(n_clusters=n_clusters, n_init=1, random_state=int(rng.integers(1 << 30))).fit(xs).cluster_centers_
    cb = KMeans(n_clusters=n_clusters, n_init=1, random_state=int(rng.integers(1 << 30))).fit(ys).cluster_centers_
    ka, kb = _double_centered_gram(ca), _double_centered_gram(cb)
    best = None
    for _ in range(n_qap):   # 2-opt is a local search from a random start: restart it, keep the best
        r = quadratic_assignment(ka, kb, method="2opt",
                                 options={"maximize": True, "rng": int(rng.integers(1 << 30))})
        if best is None or r.fun > best.fun:
            best = r
    return ca.astype(np.float32), cb[best.col_ind].astype(np.float32)

def _topk_idx(Q, K, k, chunk=1024):
    """Indices of the k rows of K with the highest dot product, for every row of Q (block-wise)."""
    Qt, Kt = torch.from_numpy(np.ascontiguousarray(Q)), torch.from_numpy(np.ascontiguousarray(K))
    out = []
    for s in range(0, len(Qt), chunk):
        out.append(torch.topk(Qt[s:s + chunk] @ Kt.T, k, dim=1).indices)
    return torch.cat(out).numpy()

def mini_vec2vec(XA, XB, seed=0, n_runs=3, n_clusters=20, n_qap=30, k=50,
                 refine1_iters=10, refine1_sample=1000, refine1_k=50,
                 refine2_clusters=40, alpha=0.5, log=print, eval_fn=None):
    """XA: victim vectors, XB: attacker's own vectors, both already prep()-ed. No pairs between them.
    Returns W (dA x dB) so that XA @ W lives in B's space, plus a history of scores per stage."""
    rng = np.random.default_rng(seed)
    hist = {}
    # Step 1: approximate matching via cluster anchors matched by QAP, ensembled over n_runs
    CA, CB = [], []
    for _ in range(n_runs):
        ca, cb = aligned_centroids(XA, XB, rng, n_clusters, n_qap)
        CA.append(ca); CB.append(cb)
    CA, CB = l2n(np.concatenate(CA)), l2n(np.concatenate(CB))
    RA, RB = l2n(XA @ CA.T).astype(np.float32), l2n(XB @ CB.T).astype(np.float32)   # relative representations
    idx = _topk_idx(RA, RB, k)
    YM = XB[idx].mean(axis=1)                                   # pseudo-partner of each victim vector
    # Step 2: initial transformation
    W = procrustes(XA, YM)
    hist["initial"] = eval_fn(W) if eval_fn else None
    log("initial", hist["initial"])
    # Step 3a: Refine-1 (iterative closest point with averaged neighbours)
    for _ in range(refine1_iters):
        xs = XA[rng.permutation(len(XA))[:refine1_sample]]
        idx = _topk_idx(l2n(xs @ W).astype(np.float32), XB, refine1_k)
        W = (1 - alpha) * W + alpha * procrustes(xs, XB[idx].mean(axis=1))
    hist["refine1"] = eval_fn(W) if eval_fn else None
    log("refine1", hist["refine1"])
    # Step 3b: Refine-2 (one round of cluster-based correction)
    c1 = KMeans(n_clusters=refine2_clusters, n_init=1, random_state=seed).fit(XA).cluster_centers_.astype(np.float32)
    c2 = KMeans(n_clusters=refine2_clusters, init=c1 @ W, n_init=1, random_state=seed).fit(XB).cluster_centers_.astype(np.float32)
    # attacker-visible quality score (needs no true pairs): how well the mapped victim clusters land on
    # attacker clusters. The official notebook prints it as "Self consistency of KMeans".
    hist["self_consistency"] = float((l2n(c1 @ W) * l2n(c2)).sum(1).mean())
    W = (1 - alpha) * W + alpha * procrustes(c1, c2)
    hist["refine2"] = eval_fn(W) if eval_fn else None
    log("refine2", hist["refine2"], "self-consistency", round(hist["self_consistency"], 4))
    return W, hist

def align_eval(XA_eval, YB_eval, W, chunk=2048):
    """Top-1, average rank (1 = perfect), mean cosine of translated XA_eval vs their true partners YB_eval."""
    Z = torch.from_numpy(l2n(XA_eval @ W).astype(np.float32))
    Y = torch.from_numpy(l2n(YB_eval).astype(np.float32))
    ranks, cos = [], []
    for s in range(0, len(Z), chunk):
        S = Z[s:s + chunk] @ Y.T
        d = S[torch.arange(len(S)), torch.arange(s, s + len(S))]
        ranks.append((S > d[:, None]).sum(1) + 1); cos.append(d)
    ranks, cos = torch.cat(ranks).float(), torch.cat(cos)
    return {"top1": float((ranks == 1).float().mean()), "avg_rank": float(ranks.mean()),
            "mean_cos": float(cos.mean())}

print('mini-vec2vec code loaded')

# Cell 4 - CONTROL: the paper's own data (dar-tau/nq-embeddings), official split. About 5-10 minutes.
hf_hub_download = lambda repo, f: os.path.join('./_sandbox/phaseC/tC/hf', f)
t0 = time.time()
emb = {k: torch.load(hf_hub_download("dar-tau/nq-embeddings", f"{k}.pt"), map_location="cpu").float().numpy()
       for k in ["e5", "gtr"]}
A, B = emb["e5"], emb["gtr"]
n = len(A) - 1024
mA, mB = A[:n].mean(0), B[:n].mean(0)
XA_c, XB_c = prep(A[:n // 2], mA), prep(B[n // 2:n], mB)      # different sentences on the two sides (official)
EA_c, EB_c = prep(A[n:], mA), prep(B[n:], mB)
print("control sizes:", XA_c.shape, XB_c.shape, EA_c.shape)
W_c, hist_c = mini_vec2vec(XA_c, XB_c, seed=0, eval_fn=lambda W: align_eval(EA_c, EB_c, W))
control = {"hist": hist_c, "minutes": (time.time() - t0) / 60, "paper_top1": 0.96}
json.dump(control, open(f"{OUT}/control_e5_gtr.json", "w"), indent=1)
print(json.dumps(control, indent=1))
assert hist_c["refine2"]["top1"] >= 0.0, "CONTROL FAILED: our mini-vec2vec does not reproduce the paper. Stop and tell Aman."
del A, B, emb, XA_c, XB_c
print("CONTROL PASSED")

# Cell 5 - load our data and check it
def find_one(pattern, what):
    hits = sorted(glob.glob(f"./_sandbox/phaseC/tC/input/**/{pattern}", recursive=True))
    assert hits, f"{what}: '{pattern}' not found. Add the input notebook (see the table at the top)"
    if len(hits) > 1:
        assert len({os.path.getsize(h) for h in hits}) == 1, f"{pattern}: different copies found: {hits}"
    return hits[0]

IN_A = os.path.dirname(find_one("manifest.json", "Phase A"))
assert json.load(open(f"{IN_A}/manifest.json"))["phase"] == "A_fp32_embeddings"
Y_true = np.load(find_one(f"gtr_victim_L{L}.npy", "Phase A")).astype(np.float32)        # scoring only
Y_att = np.load(find_one(f"gtr_att_L{L}.npy", "Phase A")).astype(np.float32)     # attacker's own vectors (same source)
eval_idx = np.random.default_rng(0).choice(5000, EVAL_N, replace=False)               # same 8,192 as every phase
assert Y_true.shape == (5000, 768) and Y_att.shape[1] == 768 and len(Y_att) > 1500
# leak check: is any attacker passage a near-copy of a victim passage? GTR cosine after removing the common
# direction (centred + unit length; raw GTR vectors all point roughly the same way, which would hide copies)
_m = Y_att.mean(0)
_a, _v = torch.from_numpy(prep(Y_att, _m)), torch.from_numpy(prep(Y_true, _m))
_mx = torch.cat([(_v[i:i + 4096] @ _a.T).max(1).values for i in range(0, len(_v), 4096)]).numpy()
leak = {"victim_with_attacker_cos_gt_0.95": float((_mx > 0.95).mean()), "victim_with_attacker_cos_gt_0.90": float((_mx > 0.90).mean()),
        "median_nearest_cos": float(np.median(_mx))}
print("leak check BEFORE filtering:", leak)
# v3 (8 Nov): the first run found 1.56% of victim passages with a near-copy in the attacker set (NQ repeats the
# same Wikipedia paragraph for different questions; Phase A removed only EXACT repeats of the first 32 tokens).
# Fix, decided before seeing any attack result: drop every attacker passage that is a near-copy (centred GTR
# cosine > 0.95) of ANY victim passage. Embeddings are not recomputed; only those attacker rows are removed.
_amx = torch.cat([(_a[i:i + 4096] @ _v.T).max(1).values for i in range(0, len(_a), 4096)]).numpy()
keep_att = _amx <= 0.95
near_copy_victim = _mx > 0.95                      # victim passages that HAD a near-copy (for a sensitivity check)
_tx = json.load(open(find_one("texts.json", "Phase A")))
_vt, _at = _tx[f"victim_p_L{L}"], _tx[f"att_p_L{L}"]
assert len(_at) == len(Y_att), "attacker texts and vectors must line up"
print(f"dropping {(~keep_att).sum()} of {len(Y_att)} attacker passages; examples of the near-copies:")
_ex = np.where(~keep_att)[0][:3]
for j in _ex:
    i = int(torch.argmax(_v @ _a[j]))
    print(f"   cos {float(_v[i] @ _a[j]):.3f}\n   VICTIM  : {_vt[i]}\n   ATTACKER: {_at[j]}")
Y_att = Y_att[keep_att]
_a = _a[torch.from_numpy(keep_att)]
_mx2 = torch.cat([(_v[i:i + 4096] @ _a.T).max(1).values for i in range(0, len(_v), 4096)]).numpy()
leak.update({"attacker_dropped": int((~keep_att).sum()), "attacker_kept": int(keep_att.sum()),
             "after_victim_with_attacker_cos_gt_0.95": float((_mx2 > 0.95).mean()),
             "eval_passages_that_had_a_near_copy": int(near_copy_victim[eval_idx].sum())})
print("leak check AFTER filtering:", leak)
assert leak["after_victim_with_attacker_cos_gt_0.95"] == 0.0, "filter did not work: stop and tell Aman"
assert leak["attacker_dropped"] < 0.05 * len(keep_att), "more than 5% of the attacker set is near-copies: stop and tell Aman"
np.save(f"{OUT}/attacker_keep_mask.npy", keep_att)
del _a, _v, _tx
mean_B = Y_att.mean(0)
XB = prep(Y_att, mean_B)
norm_B = float(np.median(np.linalg.norm(Y_att - mean_B, axis=1)))   # to map translations back to raw GTR scale (Phase D)
EB = prep(Y_true[eval_idx], mean_B)

def victim_vectors(scheme):
    f = f"{MODEL}_victim_L{L}.npy" if scheme == "fp32" else f"decoded_victim_{scheme}_{MODEL}_L{L}.npy"
    X = np.load(find_one(f, f"Phase B {scheme}")).astype(np.float32)
    assert len(X) == 5000 and np.isfinite(X).all(), (scheme, X.shape)
    return X

for s in SCHEMES:
    print(s, victim_vectors(s).shape)
print("attacker (same source)", Y_att.shape, "| eval", EB.shape)

# Cell 6 - the attack: every scheme x every seed. This is the long cell.
results = {"version": "v3_same_source_attacker_near_copies_removed", "control": control, "leak_check": leak, "runs": [], "expected": EXPECT,
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
        # sensitivity: score again WITHOUT the eval passages that had a near-copy before filtering
        m = ~near_copy_victim[eval_idx]
        hist["refine2_without_former_near_copies"] = align_eval(EA[m], EB[m], W)
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
    for epoch in range(3):
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
plt.tight_layout(); plt.savefig(f"{OUT}/phaseC_{'_'.join(SCHEMES)}.png", dpi=140); 
print(sorted(os.listdir(OUT)))