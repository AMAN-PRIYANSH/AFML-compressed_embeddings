import nbformat as nbf
from pathlib import Path
HERE = Path(__file__).parent

def write(cells, name):
    nb = nbf.v4.new_notebook(); nb["cells"] = cells
    nb["metadata"] = {"kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"},
                      "language_info": {"name": "python"}}
    nbf.write(nb, HERE / name); print("written", name, len(cells), "cells")

M = lambda s: nbf.v4.new_markdown_cell(s.strip())
C = lambda s: nbf.v4.new_code_cell(s.strip())

FIND = r'''
def find_one(pattern, what):
    hits = sorted(glob.glob(f"/kaggle/input/**/{pattern}", recursive=True))
    assert hits, f"{what}: '{pattern}' not found. Add the input notebook (see the table at the top)"
    if len(hits) > 1:
        assert len({os.path.getsize(h) for h in hits}) == 1, f"{pattern}: different copies found: {hits}"
    return hits[0]
def l2n(X): return X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)
'''

METRICS = r'''
from collections import Counter
def toks(s): return re.findall(r"\w+", s.lower())
def token_f1(p, t):
    P, T = toks(p), toks(t)
    if not P or not T: return 0.0
    c = sum((Counter(P) & Counter(T)).values())
    if c == 0: return 0.0
    pr, rc = c / len(P), c / len(T)
    return 2 * pr * rc / (pr + rc)
def ent_hits(p, ents):
    # identical rule to Phase D: whole-word, case-insensitive match of each named entity of the real passage
    pl = p.lower()
    return sum(bool(re.search(r"\b" + re.escape(e) + r"\b", pl)) for e in ents)
'''

# --------------------------------------------------------------------------------------------- E0 statistics
e0 = [M("""
# PhaseE0_STATS: are the differences between schemes real? (paired statistics on Phase C2 + D)

Neural Nexus, vector-DB compression project (team leader: Aman).

| | |
|---|---|
| **Time** | about 15 minutes |
| **Settings** | Accelerator = **None (CPU)**, Internet = **On** |
| **Inputs** | `PhaseD_INVERSION`, `phaseA_fp32_embeddings`, and all six `PhaseC2_*` notebooks |
| **Nothing to change** | |
| **Output** | `phaseE0/`: results_E0.json, stats_inversion.csv, stats_translation.csv, 2 plots |

What it does:
1. Phase C: per scheme, mean / spread over the 5 seeds, and the top-1 of the seed an attacker would pick without
   answers (highest self-consistency).
2. Phase D: every scheme was inverted on the SAME 1,000 passages, so differences can be tested passage by passage
   (paired). For each scheme vs fp32 and vs the random floor: mean difference in token F1, sentence BLEU and
   entity recovery, a 95% bootstrap interval (10,000 resamples) and a paired permutation p-value, Holm-corrected.
3. Limitation printed in the output: this measures differences between passages for ONE seed per scheme; it does
   not include seed-to-seed variation.
"""),
C(r"""
!pip install -q sacrebleu spacy
!python -m spacy download en_core_web_sm -q
"""),
C(r"""
import os, glob, json, re, numpy as np, pandas as pd
""" + FIND + METRICS + r"""
OUT = "/kaggle/working/phaseE0"; os.makedirs(OUT, exist_ok=True)
SCHEMES = ["fp32", "int8", "int4", "pq128x8", "binary", "svd256"]
EXPECT = {
    "int8 / int4 vs fp32": "token F1 difference near 0, interval includes 0",
    "pq128x8 / binary vs fp32": "about -0.06 to -0.07 token F1, interval excludes 0",
    "svd256 vs fp32": "about +0.01; may or may not exclude 0 (Phase D intervals barely touched)",
    "every scheme vs floor": "clearly above the floor (token F1 +0.07 or more, p < 0.001)",
}
for k, v in EXPECT.items(): print(f"- {k}: {v}")
"""),
C(r"""
# Phase C: seeds and the attacker-picked seed
rows = []
for s in SCHEMES:
    R = json.load(open(find_one(f"results_C_{s}.json", f"PhaseC2_{s.upper()}")))
    runs = [r for r in R["runs"] if r["scheme"] == s]
    t1 = np.array([r["refine2"]["top1"] for r in runs]); sc = np.array([r["self_consistency"] for r in runs])
    pk = runs[int(sc.argmax())]
    rows.append({"scheme": s, "seeds": len(runs), "top1_mean": t1.mean(), "top1_std": t1.std(ddof=1), "top1_min": t1.min(),
                 "top1_max": t1.max(), "picked_seed": pk["seed"], "picked_top1": pk["refine2"]["top1"],
                 "picked_cos": pk["refine2"]["mean_cos"], "picked_is_best_seed": bool(pk["refine2"]["top1"] == t1.max()),
                 "phaseD_pick_matches": R["phaseD_pick"][s]["seed"] == pk["seed"]})
tc = pd.DataFrame(rows); print(tc.round(4).to_string(index=False)); tc.to_csv(f"{OUT}/stats_translation.csv", index=False)
assert tc["phaseD_pick_matches"].all(), "Phase D did not use the self-consistency seed: stop and tell Aman"
"""),
C(r"""
# Phase D: per-passage metrics (same 1,000 passages for every scheme)
import sacrebleu, spacy
nlp = spacy.load("en_core_web_sm")
D = {s: pd.read_csv(find_one(f"inversions_{s}.csv", "PhaseD_INVERSION")) for s in SCHEMES}
truth = D["fp32"]["true"].tolist()
assert all(D[s]["true"].tolist() == truth for s in SCHEMES), "schemes were not scored on the same passages"
texts = json.load(open(find_one("texts.json", "phaseA")))["victim_p_L32"]
eval_idx = np.random.default_rng(0).choice(50000, 8192, replace=False); pick = eval_idx[:len(truth)]
assert [texts[i] for i in pick] == truth, "passage order differs from Phase D"
other = [texts[i] for i in np.random.default_rng(2).choice(np.setdiff1d(np.arange(50000), pick), len(truth), replace=False)]
ents = [[e.text.lower() for e in d.ents] for d in nlp.pipe(truth, batch_size=256)]
n_ent = np.array([len(e) for e in ents])
def per_passage(preds):
    return {"f1": np.array([token_f1(p, t) for p, t in zip(preds, truth)]),
            "bleu": np.array([sacrebleu.sentence_bleu(p, [t]).score for p, t in zip(preds, truth)]),
            "ent": np.array([ent_hits(p, e) for p, e in zip(preds, ents)])}
PP = {s: per_passage(D[s]["recovered"].fillna("").tolist()) for s in SCHEMES}
PP["floor"] = per_passage(other)
for s in SCHEMES:   # must reproduce Phase D's own numbers
    assert abs(PP[s]["f1"].mean() - D[s]["token_f1"].mean()) < 1e-6
print({s: round(PP[s]["f1"].mean(), 4) for s in PP})
"""),
C(r"""
# paired bootstrap + sign-flip permutation test, Holm correction
rng = np.random.default_rng(0); B = 10000; n = len(truth)
idx = rng.integers(0, n, (B, n)); flips = rng.choice([-1.0, 1.0], (B, n))
def compare(a, b):
    out = {}
    for m in ["f1", "bleu"]:
        d = PP[a][m] - PP[b][m]
        boot = d[idx].mean(1)
        perm = (flips * d).mean(1)
        out[m] = {"diff": float(d.mean()), "ci95": [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))],
                  "p": float((np.abs(perm) >= abs(d.mean())).mean() + 1 / B)}
    ra = PP[a]["ent"][idx].sum(1) / n_ent[idx].sum(1); rb = PP[b]["ent"][idx].sum(1) / n_ent[idx].sum(1)
    out["entity"] = {"diff": float(PP[a]["ent"].sum() / n_ent.sum() - PP[b]["ent"].sum() / n_ent.sum()),
                     "ci95": [float(np.percentile(ra - rb, 2.5)), float(np.percentile(ra - rb, 97.5))]}
    return out
res = {f"{s}_vs_fp32": compare(s, "fp32") for s in SCHEMES if s != "fp32"}
res.update({f"{s}_vs_floor": compare(s, "floor") for s in SCHEMES})
for fam in ["_vs_fp32", "_vs_floor"]:       # Holm within each family of comparisons, token F1 p-values
    keys = sorted([k for k in res if k.endswith(fam)], key=lambda k: res[k]["f1"]["p"])
    m = len(keys); running = 0
    for i, k in enumerate(keys):
        running = max(running, min(1.0, (m - i) * res[k]["f1"]["p"])); res[k]["f1"]["p_holm"] = running
rows = [{"comparison": k, "f1_diff": v["f1"]["diff"], "f1_ci_low": v["f1"]["ci95"][0], "f1_ci_high": v["f1"]["ci95"][1],
         "f1_p_holm": v["f1"]["p_holm"], "bleu_diff": v["bleu"]["diff"], "bleu_ci_low": v["bleu"]["ci95"][0],
         "bleu_ci_high": v["bleu"]["ci95"][1], "entity_diff": v["entity"]["diff"], "entity_ci_low": v["entity"]["ci95"][0],
         "entity_ci_high": v["entity"]["ci95"][1],
         "significant_f1": v["f1"]["p_holm"] < 0.05 and not (v["f1"]["ci95"][0] <= 0 <= v["f1"]["ci95"][1])} for k, v in res.items()]
ti = pd.DataFrame(rows); print(ti.round(4).to_string(index=False)); ti.to_csv(f"{OUT}/stats_inversion.csv", index=False)
print("\nLIMITATION: differences are across passages for ONE seed per scheme (the attacker-picked one); "
      "seed-to-seed variation is shown in the Phase C table, not in these intervals.")
json.dump({"translation": tc.to_dict("records"), "inversion": res, "expected": EXPECT}, open(f"{OUT}/results_E0.json", "w"), indent=1, default=float)
"""),
C(r"""
import matplotlib.pyplot as plt
fig, ax = plt.subplots(1, 2, figsize=(13, 4.5))
means = [PP[s]["f1"].mean() for s in SCHEMES]
ci = [np.percentile(PP[s]["f1"][idx].mean(1), [2.5, 97.5]) for s in SCHEMES]
err = np.array([[m - c[0] for m, c in zip(means, ci)], [c[1] - m for m, c in zip(means, ci)]])
ax[0].bar(SCHEMES, means, yerr=err, capsize=4, color="tab:red", label="recovered from stolen vectors (95% CI)")
ax[0].axhline(PP["floor"]["f1"].mean(), color="gray", ls="--", label="random other passage (floor)")
ax[0].set_xlabel("Compression scheme"); ax[0].set_ylabel("Mean token F1 vs the real passage")
ax[0].set_title("Text recovered per scheme (1,000 passages)"); ax[0].legend(fontsize=8)
comp = [s for s in SCHEMES if s != "fp32"]
d = [res[f"{s}_vs_fp32"]["f1"]["diff"] for s in comp]
e = np.array([[x - res[f"{s}_vs_fp32"]["f1"]["ci95"][0] for x, s in zip(d, comp)],
              [res[f"{s}_vs_fp32"]["f1"]["ci95"][1] - x for x, s in zip(d, comp)]])
ax[1].bar(comp, d, yerr=e, capsize=4, color="tab:blue", label="difference vs fp32 (95% CI)")
ax[1].axhline(0, color="black", lw=1)
ax[1].set_xlabel("Compression scheme"); ax[1].set_ylabel("Token F1 difference vs fp32")
ax[1].set_title("Does compression reduce the leak? (paired, same passages)"); ax[1].legend(fontsize=8)
plt.tight_layout(); plt.savefig(f"{OUT}/phaseE0_stats.png", dpi=140); plt.show()
print(sorted(os.listdir(OUT)))
""")]
write(e0, "PhaseE0_STATS.ipynb")

# --------------------------------------------------------------------------------------- shared inverter cell
INV = r'''
import vec2text, sacrebleu, spacy
from transformers import AutoModel, AutoTokenizer
corrector = vec2text.load_pretrained_corrector("gtr-base")
gtr_tok = AutoTokenizer.from_pretrained("sentence-transformers/gtr-t5-base")
gtr_enc = AutoModel.from_pretrained("sentence-transformers/gtr-t5-base").encoder.to(DEVICE).eval()
nlp = spacy.load("en_core_web_sm")
''' + METRICS + r'''
@torch.no_grad()
def embed_gtr(batch_texts, bs=128):
    # identical to Phase A / D: mean-pooled T5 encoder output, no dense layer, not normalised
    out = []
    for i in range(0, len(batch_texts), bs):
        b = gtr_tok(batch_texts[i:i + bs], return_tensors="pt", max_length=128, truncation=True,
                    padding="max_length").to(DEVICE)
        h = gtr_enc(input_ids=b["input_ids"], attention_mask=b["attention_mask"]).last_hidden_state
        out.append(vec2text.models.model_utils.mean_pool(h, b["attention_mask"]).float().cpu())
    return torch.cat(out).numpy()

def invert(V, steps=20, beam=0, batch=100):
    preds = []
    for i in range(0, len(V), batch):
        preds += vec2text.invert_embeddings(embeddings=torch.tensor(np.asarray(V[i:i + batch], dtype=np.float32), device=DEVICE),
                                            corrector=corrector, num_steps=steps, sequence_beam_width=beam)
    return preds

truth_ents = [[e.text.lower() for e in d.ents] for d in nlp.pipe(truth, batch_size=256)]
def score(preds, name):
    f1 = np.array([token_f1(p, t) for p, t in zip(preds, truth)])
    cos = (l2n(embed_gtr(preds)) * l2n(Y_true[pick])).sum(1)
    r = {"name": name, "n": len(preds), "bleu": sacrebleu.corpus_bleu(preds, [truth]).score, "token_f1": float(f1.mean()),
         "exact_match": float(np.mean([p.strip() == t.strip() for p, t in zip(preds, truth)])),
         "entity_recovery": sum(ent_hits(p, e) for p, e in zip(preds, truth_ents)) / max(sum(map(len, truth_ents)), 1),
         "gtr_cos_recovered_vs_true": float(cos.mean())}
    print(json.dumps(r)); return r, f1
'''

# --------------------------------------------------------------------------------- E1 supervised upper bound
e1 = [M("""
# PhaseE1_SUPERVISED: what would an attacker WITH pairs recover? (upper bound for a linear / small-MLP attack)

Neural Nexus, vector-DB compression project (team leader: Aman).

| | |
|---|---|
| **Time** | about 45 minutes |
| **Settings** | Accelerator = **GPU T4** (x2 is fine, one is used), Internet = **On** |
| **Inputs** | `phaseA_fp32_embeddings`, `PhaseC2_FP32`, `PhaseD_INVERSION` |
| **Nothing to change** | |
| **Output** | `phaseE1/`: results_E1.json, translated_eval_sup_<map>.npy (for E2), inversions CSV, plot |

Why: Phase D found the translations are too far from the true vectors (cosine ~0.48) for vec2text to rebuild the
text. Is that the unsupervised method's fault, or does even a supervised map stay that far? Maps (fp32 victim):
- **procrustes / ridge / mlp**: trained on the attacker's OWN 18,854 passages embedded with both models (pairs);
  a realistic stronger attacker.
- **oracle_linear**: least squares fitted on the victim's own 50,000 pairs. NOT an attack (it uses the answers);
  it is the best any linear map could ever do on this data.
Each map translates the same 1,000 passages as Phase D, which are then inverted and scored the same way.
"""),
C(r"""
!pip install -q "transformers==4.46.3" "vec2text==0.0.13" sacrebleu spacy
!python -m spacy download en_core_web_sm -q
"""),
C(r"""
import os, glob, json, re, time, numpy as np, torch, pandas as pd
DEVICE = "cuda"; assert torch.cuda.is_available(), "Set Accelerator = GPU T4"
""" + FIND + r"""
OUT = "/kaggle/working/phaseE1"; os.makedirs(OUT, exist_ok=True)
L, N = 32, 1000
texts = json.load(open(find_one("texts.json", "phaseA")))["victim_p_L32"]
Y_true = np.load(find_one("gtr_victim_L32.npy", "phaseA")).astype(np.float32)
X_vic = np.load(find_one("bge-large_victim_L32.npy", "phaseA")).astype(np.float32)
X_att = np.load(find_one("bge-large_att_L32.npy", "phaseA")).astype(np.float32)
Y_att = np.load(find_one("gtr_att_L32.npy", "phaseA")).astype(np.float32)
keep = np.load(find_one("attacker_keep_mask.npy", "PhaseC2_FP32"))
mean_A = np.load(find_one("mean_A_fp32.npy", "PhaseC2_FP32")); mean_B = np.load(find_one("mean_B.npy", "PhaseC2_FP32"))
C2 = json.load(open(find_one("results_C_fp32.json", "PhaseC2_FP32"))); norm_B = C2["norm_B"]
D = json.load(open(find_one("results_D.json", "PhaseD_INVERSION")))
assert len(keep) == len(X_att) == len(Y_att) and keep.sum() == 18854
X_att, Y_att = X_att[keep], Y_att[keep]                 # same attacker passages as Phase C2 (near-copies removed)
eval_idx = np.random.default_rng(0).choice(50000, 8192, replace=False); pick = eval_idx[:N]
truth = [texts[i] for i in pick]
prep = lambda X, m: l2n(X - m).astype(np.float32)
XA, YA = prep(X_att, mean_A), prep(Y_att, mean_B)
EA, EB = prep(X_vic[eval_idx], mean_A), prep(Y_true[eval_idx], mean_B)
EXPECT = {
    "procrustes (pairs)": "top-1 ~0.99, cosine ~0.57 (Phase C v1: 0.993 / 0.565)",
    "mlp (pairs)": "top-1 ~0.99, cosine ~0.67 (Phase C v1: 0.993 / 0.670)",
    "oracle_linear": "the highest cosine of the linear maps; guess 0.6-0.7",
    "inversion": "token F1 above Phase D fp32 (0.238) but far below the true-vector control (0.957). Guess 0.25-0.40",
}
for k, v in EXPECT.items(): print(f"- {k}: {v}")
"""),
C(INV),
C(r"""
def align_eval(Z, Y):
    Z, Y = torch.from_numpy(l2n(Z).astype(np.float32)), torch.from_numpy(l2n(Y).astype(np.float32))
    S = Z @ Y.T; d = S.diag(); ranks = (S > d[:, None]).sum(1) + 1
    return {"top1": float((ranks == 1).float().mean()), "avg_rank": float(ranks.float().mean()), "mean_cos": float(d.mean())}
maps = {}
U, _, Vt = np.linalg.svd(XA.T.astype(np.float64) @ YA.astype(np.float64), full_matrices=False)
maps["procrustes"] = lambda Z, W=(U @ Vt).astype(np.float32): Z @ W
Wr = np.linalg.solve(XA.T @ XA + 1e-2 * np.eye(XA.shape[1]), XA.T @ YA).astype(np.float32)
maps["ridge"] = lambda Z, W=Wr: Z @ W
XV, YV = prep(X_vic, mean_A), prep(Y_true, mean_B)
Wo = np.linalg.solve(XV.T @ XV + 1e-2 * np.eye(XV.shape[1]), XV.T @ YV).astype(np.float32)
maps["oracle_linear"] = lambda Z, W=Wo: Z @ W
torch.manual_seed(0)
mlp = torch.nn.Sequential(torch.nn.Linear(1024, 2048), torch.nn.GELU(), torch.nn.Linear(2048, 768)).to(DEVICE)
opt = torch.optim.AdamW(mlp.parameters(), lr=1e-3, weight_decay=1e-4)
Xt, Yt = torch.from_numpy(XA).to(DEVICE), torch.from_numpy(YA).to(DEVICE)
for epoch in range(30):          # same recipe as Phase C v1
    perm = torch.randperm(len(Xt), device=DEVICE)
    for i in range(0, len(Xt), 256):
        b = perm[i:i + 256]
        loss = 1 - torch.nn.functional.cosine_similarity(mlp(Xt[b]), Yt[b]).mean()
        opt.zero_grad(); loss.backward(); opt.step()
mlp.eval()
maps["mlp"] = lambda Z: mlp(torch.from_numpy(Z).to(DEVICE)).detach().cpu().numpy()
results = {"n": N, "expected": EXPECT, "maps": {}, "phaseD_fp32": D["schemes"]["fp32"],
           "phaseD_positive_control": D["controls"]["positive_true_vectors"]}
for name, f in maps.items():
    with torch.no_grad():
        Z = f(EA)
    results["maps"][name] = {"translation": align_eval(Z, EB)}
    np.save(f"{OUT}/translated_eval_sup_{name}.npy", (l2n(Z) * norm_B + mean_B).astype(np.float32))
    print(name, results["maps"][name]["translation"])
"""),
C(r"""
# invert the first 1,000 translations of each map (same passages and settings as Phase D)
csv = {"true": truth}
for name in maps:
    t0 = time.time()
    V = np.load(f"{OUT}/translated_eval_sup_{name}.npy")[:N]
    preds = invert(V)
    r, f1 = score(preds, name)
    r["minutes"] = (time.time() - t0) / 60
    results["maps"][name]["inversion"] = r; csv[name] = preds; csv[f"{name}_f1"] = f1
    for j in range(2): print("   TRUE:", truth[j], "\n   GOT :", preds[j])
    json.dump(results, open(f"{OUT}/results_E1.json", "w"), indent=1)
pd.DataFrame(csv).to_csv(f"{OUT}/inversions_supervised.csv", index=False)
rows = [{"map": "mini-vec2vec (Phase D, no pairs)", "cos": None, "token_f1": D["schemes"]["fp32"]["token_f1"],
         "bleu": D["schemes"]["fp32"]["bleu"], "entities": D["schemes"]["fp32"]["entity_recovery"]}]
rows += [{"map": k, "cos": v["translation"]["mean_cos"], "token_f1": v["inversion"]["token_f1"], "bleu": v["inversion"]["bleu"],
          "entities": v["inversion"]["entity_recovery"]} for k, v in results["maps"].items()]
df = pd.DataFrame(rows); print(df.round(3).to_string(index=False))
"""),
C(r"""
import matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize=(8, 4.5))
names = ["mini-vec2vec\n(no pairs)"] + list(results["maps"])
vals = [D["schemes"]["fp32"]["token_f1"]] + [results["maps"][k]["inversion"]["token_f1"] for k in results["maps"]]
ax.bar(names, vals, color=["tab:red"] + ["tab:orange"] * 3 + ["tab:gray"], label="token F1 of recovered text")
ax.axhline(D["controls"]["positive_true_vectors"]["token_f1"], color="tab:green", ls="--", label="true vectors (upper bound)")
ax.axhline(D["controls"]["floor_random_other_passage"]["token_f1"], color="gray", ls=":", label="random other passage (floor)")
ax.set_xlabel("How the stolen fp32 vector was translated into GTR space"); ax.set_ylabel("Token F1 vs the real passage")
ax.set_title("Text recovered: unsupervised vs supervised translation"); ax.legend(fontsize=8); ax.set_ylim(0, 1)
plt.tight_layout(); plt.savefig(f"{OUT}/phaseE1_supervised.png", dpi=140); plt.show()
json.dump(results, open(f"{OUT}/results_E1.json", "w"), indent=1)
print(sorted(os.listdir(OUT)))
""")]
write(e1, "PhaseE1_SUPERVISED.ipynb")

# ----------------------------------------------------------------------------------- E2 stronger decoding
e2 = [M("""
# PhaseE2_DECODING: does a stronger inverter recover more? (more correction steps, beam search)

Neural Nexus, vector-DB compression project (team leader: Aman).

| | |
|---|---|
| **Time** | about 2-3 hours (beam search is slow) |
| **Settings** | Accelerator = **GPU T4** (x2 is fine), Internet = **On** |
| **Inputs** | `phaseA_fp32_embeddings`, `PhaseC2_FP32`, `PhaseC2_PQ128X8`, `PhaseE1_SUPERVISED` |
| **Run after** | PhaseE1_SUPERVISED has finished |
| **Output** | `phaseE2/`: results_E2.json, inversions CSV, plot |

200 passages (the first 200 of Phase D's 1,000). Four vector sets: true GTR vectors (control), mini-vec2vec fp32,
mini-vec2vec pq128x8, supervised MLP (from E1). Three inverter settings: 20 steps (Phase D setting),
50 steps, 50 steps + beam width 4 (vec2text paper's strongest).
"""),
C(r"""
!pip install -q "transformers==4.46.3" "vec2text==0.0.13" sacrebleu spacy
!python -m spacy download en_core_web_sm -q
"""),
C(r"""
import os, glob, json, re, time, numpy as np, torch, pandas as pd
DEVICE = "cuda"; assert torch.cuda.is_available(), "Set Accelerator = GPU T4"
""" + FIND + r"""
OUT = "/kaggle/working/phaseE2"; os.makedirs(OUT, exist_ok=True)
N = 200
texts = json.load(open(find_one("texts.json", "phaseA")))["victim_p_L32"]
Y_true = np.load(find_one("gtr_victim_L32.npy", "phaseA")).astype(np.float32)
eval_idx = np.random.default_rng(0).choice(50000, 8192, replace=False); pick = eval_idx[:N]
truth = [texts[i] for i in pick]
SETS = {"true_vectors": Y_true[pick],
        "minivec2vec_fp32": np.load(find_one("translated_eval_fp32.npy", "PhaseC2_FP32"))[:N],
        "minivec2vec_pq128x8": np.load(find_one("translated_eval_pq128x8.npy", "PhaseC2_PQ128X8"))[:N],
        "supervised_mlp": np.load(find_one("translated_eval_sup_mlp.npy", "PhaseE1_SUPERVISED"))[:N]}
CONFIGS = [("20 steps", 20, 0, 100), ("50 steps", 50, 0, 100), ("50 steps + beam 4", 50, 4, 20)]
EXPECT = {
    "true vectors": "BLEU >= 85 at 20 steps; higher with 50 steps + beam (vec2text paper ~97). GATE: 20-step BLEU > 75",
    "translated sets": "small gain at most (token F1 +0.00 to +0.03): the translation, not the decoder, is the bottleneck",
}
for k, v in EXPECT.items(): print(f"- {k}: {v}")
"""),
C(INV),
C(r"""
results = {"n": N, "expected": EXPECT, "runs": []}; csv = {"true": truth}
for set_name, V in SETS.items():
    for cfg, steps, beam, bs in CONFIGS:
        t0 = time.time()
        preds = invert(V, steps=steps, beam=beam, batch=bs)
        r, f1 = score(preds, f"{set_name} | {cfg}")
        r.update({"set": set_name, "config": cfg, "minutes": (time.time() - t0) / 60}); results["runs"].append(r)
        csv[f"{set_name} | {cfg}"] = preds
        print(f"  ({r['minutes']:.1f} min)")
        if set_name == "true_vectors" and cfg == "20 steps":
            assert r["bleu"] > 75, "Inverter control failed: stop and tell Aman"
        json.dump(results, open(f"{OUT}/results_E2.json", "w"), indent=1)
pd.DataFrame(csv).to_csv(f"{OUT}/inversions_decoding.csv", index=False)
df = pd.DataFrame(results["runs"]).pivot(index="set", columns="config", values="token_f1")
print(df.round(3))
"""),
C(r"""
import matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize=(9, 4.5)); w = 0.25
sets = list(SETS); x = np.arange(len(sets))
for i, (cfg, *_ ) in enumerate(CONFIGS):
    ax.bar(x + (i - 1) * w, [df.loc[s, cfg] for s in sets], w, label=cfg)
ax.set_xticks(x); ax.set_xticklabels(sets, rotation=10)
ax.set_xlabel("Vectors given to the inverter"); ax.set_ylabel("Token F1 vs the real passage (200 passages)")
ax.set_title("Does a stronger inverter recover more text?"); ax.legend(); ax.set_ylim(0, 1)
plt.tight_layout(); plt.savefig(f"{OUT}/phaseE2_decoding.png", dpi=140); plt.show()
print(sorted(os.listdir(OUT)))
""")]
write(e2, "PhaseE2_DECODING.ipynb")
