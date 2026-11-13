import os, matplotlib; matplotlib.use('Agg')


import types, re, numpy as np, torch
class _Ent:
    def __init__(s, t): s.text = t
class _Doc:
    def __init__(s, t): s.ents = [_Ent(w) for w in t.split() if w[:1].isupper()]
class _NLP:
    def pipe(s, ts, batch_size=1): return [_Doc(t) for t in ts]
spacy = types.SimpleNamespace(load=lambda n: _NLP())


import os, glob, json, re, numpy as np, pandas as pd

def find_one(pattern, what):
    hits = sorted(glob.glob(f"./_sandbox/phaseE/tE/in/**/{pattern}", recursive=True))
    assert hits, f"{what}: '{pattern}' not found. Add the input notebook (see the table at the top)"
    if len(hits) > 1:
        assert len({os.path.getsize(h) for h in hits}) == 1, f"{pattern}: different copies found: {hits}"
    return hits[0]
def l2n(X): return X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)

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

OUT = "./_sandbox/phaseE/tE/work/phaseE0"; os.makedirs(OUT, exist_ok=True)
SCHEMES = ["fp32", "int8", "int4", "pq128x8", "binary", "svd256"]
EXPECT = {
    "int8 / int4 vs fp32": "token F1 difference near 0, interval includes 0",
    "pq128x8 / binary vs fp32": "about -0.06 to -0.07 token F1, interval excludes 0",
    "svd256 vs fp32": "about +0.01; may or may not exclude 0 (Phase D intervals barely touched)",
    "every scheme vs floor": "clearly above the floor (token F1 +0.07 or more, p < 0.001)",
}
for k, v in EXPECT.items(): print(f"- {k}: {v}")

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

# Phase D: per-passage metrics (same 1,000 passages for every scheme)
import sacrebleu
nlp = spacy.load("en_core_web_sm")
D = {s: pd.read_csv(find_one(f"inversions_{s}.csv", "PhaseD_INVERSION")) for s in SCHEMES}
truth = D["fp32"]["true"].tolist()
assert all(D[s]["true"].tolist() == truth for s in SCHEMES), "schemes were not scored on the same passages"
texts = json.load(open(find_one("texts.json", "phaseA")))["victim_p_L32"]
eval_idx = np.random.default_rng(0).choice(5000, 1024, replace=False); pick = eval_idx[:len(truth)]
assert [texts[i] for i in pick] == truth, "passage order differs from Phase D"
other = [texts[i] for i in np.random.default_rng(2).choice(np.setdiff1d(np.arange(5000), pick), len(truth), replace=False)]
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
plt.tight_layout(); plt.savefig(f"{OUT}/phaseE0_stats.png", dpi=140); 
print(sorted(os.listdir(OUT)))