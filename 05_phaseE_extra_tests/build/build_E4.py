import sys
sys.path.insert(0, ".")
from build_E import write, M, C, FIND, METRICS, INV   # noqa: E402  (re-uses the shared cells; also rebuilds E0-E2 files)

e4 = [M("""
# PhaseE4_NONORTHO: can an attacker WITHOUT pairs get the supervised gain? (non-orthogonal refinement)

Neural Nexus, vector-DB compression project (team leader: Aman).

| | |
|---|---|
| **Time** | about 40 minutes |
| **Settings** | Accelerator = **GPU T4** (x2 is fine), Internet = **On** |
| **Inputs** | `phaseA_fp32_embeddings`, `PHASE_B_pq128x8` (amanpriyansh/phase-b-pq128x8), `PhaseC2_FP32`, `PhaseC2_PQ128X8`, `PhaseD_INVERSION`, `PhaseE1_SUPERVISED` |
| **Nothing to change** | |
| **Output** | `phaseE4/`: results_E4.json, inversions CSV, entity re-scoring table, plot |

Why: Phase E1 showed that, WITH pairs, an orthogonal map (Procrustes, the kind mini-vec2vec uses) reaches cosine
0.57 and token F1 0.28, while an unconstrained linear map (ridge) reaches 0.67 and 0.40. mini-vec2vec already gets
cosine 0.51 without pairs, close to its own orthogonal ceiling. So the limit may be the orthogonal constraint, not
the lack of pairs. This notebook keeps the attacker unsupervised and only replaces the last step:
1. Start from mini-vec2vec's map (the seed picked by self-consistency in Phase C2).
2. For every stolen vector, take the mean of its 10 nearest attacker GTR vectors as a pseudo-target (the attacker
   has these; no victim text, no pairs).
3. Fit a ridge map stolen -> pseudo-target over all 50,000 stolen vectors. Repeat steps 2-3 five times.
The setting (k = 10, 5 rounds, ridge 0.01) is fixed BEFORE running; all rounds are logged, but only round 5 is
inverted, so the true answers never choose the result. Schemes: fp32 and pq128x8.
Also: re-scores the named entities of Phase D, E1 and this notebook with ONE pinned spaCy model (3.8.0), because
Phase E0 and E1 fell back to Kaggle's preinstalled spaCy model when the 3.8.0 download timed out.
"""),
C(r"""
import subprocess, sys
!pip install -q "transformers==4.46.3" "vec2text==0.0.13" sacrebleu spacy
for attempt in range(5):   # the model download timed out once (HTTP 504) in E0/E1: retry
    r = subprocess.run([sys.executable, "-m", "spacy", "download", "en_core_web_sm-3.8.0", "--direct", "-q"])
    if r.returncode == 0: break
import spacy, importlib.metadata as md
print("spaCy", spacy.__version__, "| en_core_web_sm", md.version("en_core_web_sm"))
assert md.version("en_core_web_sm") == "3.8.0", "could not install the pinned spaCy model: stop and tell Aman"
"""),
C(r"""
import os, glob, json, re, time, numpy as np, torch, pandas as pd
DEVICE = "cuda"; assert torch.cuda.is_available(), "Set Accelerator = GPU T4"
""" + FIND + r"""
OUT = "/kaggle/working/phaseE4"; os.makedirs(OUT, exist_ok=True)
N, K, ROUNDS, LAM = 1000, 10, 5, 1e-2
texts = json.load(open(find_one("texts.json", "phaseA")))["victim_p_L32"]
Y_true = np.load(find_one("gtr_victim_L32.npy", "phaseA")).astype(np.float32)
Y_att = np.load(find_one("gtr_att_L32.npy", "phaseA")).astype(np.float32)
keep = np.load(find_one("attacker_keep_mask.npy", "PhaseC2_FP32")); Y_att = Y_att[keep]
mean_B = np.load(find_one("mean_B.npy", "PhaseC2_FP32"))
norm_B = json.load(open(find_one("results_C_fp32.json", "PhaseC2_FP32")))["norm_B"]
eval_idx = np.random.default_rng(0).choice(50000, 8192, replace=False); pick = eval_idx[:N]
truth = [texts[i] for i in pick]
prep = lambda X, m: l2n(X - m).astype(np.float32)
XB = prep(Y_att, mean_B); EB = prep(Y_true[eval_idx], mean_B)
VICTIM = {"fp32": "bge-large_victim_L32.npy", "pq128x8": "decoded_victim_pq128x8_bge-large_L32.npy"}
EXPECT = {
    "round 0": "= Phase C2 picked seed: fp32 top-1 0.951 / cosine 0.508; pq128x8 0.758 / 0.402",
    "round 5 cosine": "higher than round 0 but below supervised ridge (0.665). Guess fp32 0.53-0.62",
    "round 5 token F1": "between Phase D (0.238) and supervised ridge (0.399). Guess fp32 0.25-0.33",
    "risk": "top-1 may DROP while cosine rises (ridge targets are neighbour averages); we report both",
}
for k, v in EXPECT.items(): print(f"- {k}: {v}")
"""),
C(INV),
C(r"""
def align_eval(Z, Y):
    Z, Y = torch.from_numpy(l2n(Z).astype(np.float32)), torch.from_numpy(l2n(Y).astype(np.float32))
    ranks, cos = [], []
    for s in range(0, len(Z), 2048):
        S = Z[s:s + 2048] @ Y.T; d = S[torch.arange(len(S)), torch.arange(s, s + len(S))]
        ranks.append((S > d[:, None]).sum(1) + 1); cos.append(d)
    ranks = torch.cat(ranks).float()
    return {"top1": float((ranks == 1).float().mean()), "avg_rank": float(ranks.mean()), "mean_cos": float(torch.cat(cos).mean())}
XBt = torch.from_numpy(XB).to(DEVICE)
def knn_mean(Z, k):
    Zt = torch.from_numpy(l2n(Z).astype(np.float32)).to(DEVICE); out = []
    for s in range(0, len(Zt), 4096):
        idx = torch.topk(Zt[s:s + 4096] @ XBt.T, k, dim=1).indices
        out.append(XBt[idx].mean(1).cpu())
    return torch.cat(out).numpy()

results = {"setting": {"k": K, "rounds": ROUNDS, "ridge": LAM}, "expected": EXPECT, "schemes": {}}
for scheme, f in VICTIM.items():
    R = json.load(open(find_one(f"results_C_{scheme}.json", f"PhaseC2_{scheme.upper()}")))
    seed = R["phaseD_pick"][scheme]["seed"]
    W = np.load(find_one(f"W_{scheme}_s{seed}.npy", f"PhaseC2_{scheme.upper()}"))
    mean_A = np.load(find_one(f"mean_A_{scheme}.npy", f"PhaseC2_{scheme.upper()}"))
    XA = prep(np.load(find_one(f, f"victim vectors {scheme}")).astype(np.float32), mean_A)
    EA = XA[eval_idx]
    hist = [{"round": 0, **align_eval(EA @ W, EB)}]; print(scheme, hist[-1])
    G = XA.T.astype(np.float64) @ XA.astype(np.float64) + LAM * np.eye(XA.shape[1])
    for rnd in range(1, ROUNDS + 1):
        T = knn_mean(XA @ W, K)                                              # pseudo-targets, attacker data only
        W = np.linalg.solve(G, XA.T.astype(np.float64) @ T.astype(np.float64)).astype(np.float32)
        hist.append({"round": rnd, **align_eval(EA @ W, EB)}); print(scheme, hist[-1])
    np.save(f"{OUT}/W_nonortho_{scheme}.npy", W)
    V = (l2n(EA[:N] @ W) * norm_B + mean_B).astype(np.float32)
    t0 = time.time(); preds = invert(V); r, f1 = score(preds, f"{scheme} non-orthogonal round {ROUNDS}")
    r["minutes"] = (time.time() - t0) / 60
    results["schemes"][scheme] = {"picked_seed": seed, "rounds": hist, "inversion": r}
    pd.DataFrame({"true": truth, "recovered": preds, "token_f1": f1}).to_csv(f"{OUT}/inversions_nonortho_{scheme}.csv", index=False)
    for j in range(2): print("   TRUE:", truth[j], "\n   GOT :", preds[j])
    json.dump(results, open(f"{OUT}/results_E4.json", "w"), indent=1)
"""),
C(r"""
# entity re-scoring with the ONE pinned spaCy model, for every inversion CSV we have
def ent_rate(preds):
    return sum(ent_hits(p, e) for p, e in zip(preds, truth_ents)) / max(sum(map(len, truth_ents)), 1)
rows = []
for s in ["fp32", "int8", "int4", "pq128x8", "binary", "svd256"]:
    d = pd.read_csv(find_one(f"inversions_{s}.csv", "PhaseD_INVERSION"))
    assert d["true"].tolist() == truth
    rows.append({"source": "Phase D mini-vec2vec", "set": s, "entity_recovery": ent_rate(d["recovered"].fillna("").tolist())})
dc = pd.read_csv(find_one("inversions_controls.csv", "PhaseD_INVERSION"))
for c in ["positive_control", "scale_control"]:
    rows.append({"source": "Phase D control", "set": c, "entity_recovery": ent_rate(dc[c].fillna("").tolist())})
de = pd.read_csv(find_one("inversions_supervised.csv", "PhaseE1_SUPERVISED"))
for c in ["procrustes", "ridge", "oracle_linear", "mlp"]:
    rows.append({"source": "E1 supervised", "set": c, "entity_recovery": ent_rate(de[c].fillna("").tolist())})
for s in VICTIM:
    d = pd.read_csv(f"{OUT}/inversions_nonortho_{s}.csv")
    rows.append({"source": "E4 non-orthogonal", "set": s, "entity_recovery": ent_rate(d["recovered"].fillna("").tolist())})
other = [texts[i] for i in np.random.default_rng(2).choice(np.setdiff1d(np.arange(50000), pick), N, replace=False)]
rows.append({"source": "floor", "set": "random other passage", "entity_recovery": ent_rate(other)})
ents = pd.DataFrame(rows); print(ents.round(4).to_string(index=False)); ents.to_csv(f"{OUT}/entities_rescored_spacy380.csv", index=False)
results["entities_rescored_spacy_3_8_0"] = rows
json.dump(results, open(f"{OUT}/results_E4.json", "w"), indent=1)
"""),
C(r"""
import matplotlib.pyplot as plt
fig, ax = plt.subplots(1, 2, figsize=(13, 4.5))
for s, col in [("fp32", "tab:blue"), ("pq128x8", "tab:orange")]:
    h = results["schemes"][s]["rounds"]
    ax[0].plot([x["round"] for x in h], [x["mean_cos"] for x in h], marker="o", color=col, label=f"{s}: cosine to true vector")
    ax[0].plot([x["round"] for x in h], [x["top1"] for x in h], marker="s", ls="--", color=col, label=f"{s}: top-1")
ax[0].axhline(0.665, color="tab:green", ls=":", label="supervised ridge cosine (E1, with pairs)")
ax[0].set_xlabel("Refinement round (0 = mini-vec2vec)"); ax[0].set_ylabel("Score on 8,192 passages")
ax[0].set_title("Non-orthogonal refinement without pairs"); ax[0].legend(fontsize=8); ax[0].set_ylim(0, 1.05)
E1 = json.load(open(find_one("results_E1.json", "PhaseE1_SUPERVISED")))
names = ["mini-vec2vec\n(Phase D)", "non-orthogonal\n(E4, no pairs)", "ridge\n(E1, pairs)"]
D = json.load(open(find_one("results_D.json", "PhaseD_INVERSION")))
vals = [D["schemes"]["fp32"]["token_f1"], results["schemes"]["fp32"]["inversion"]["token_f1"], E1["maps"]["ridge"]["inversion"]["token_f1"]]
ax[1].bar(names, vals, color=["tab:gray", "tab:blue", "tab:green"], label="fp32 token F1")
ax[1].axhline(D["controls"]["floor_random_other_passage"]["token_f1"], color="gray", ls="--", label="random other passage (floor)")
ax[1].set_xlabel("How the stolen fp32 vectors were translated"); ax[1].set_ylabel("Token F1 vs the real passage (1,000)")
ax[1].set_title("Text recovered: does the unsupervised attacker catch up?"); ax[1].legend(fontsize=8); ax[1].set_ylim(0, 1)
plt.tight_layout(); plt.savefig(f"{OUT}/phaseE4.png", dpi=140); plt.show()
print(sorted(os.listdir(OUT)))
""")]
write(e4, "PhaseE4_NONORTHO.ipynb")
