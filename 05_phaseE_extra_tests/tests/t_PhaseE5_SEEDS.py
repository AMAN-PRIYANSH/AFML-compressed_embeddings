import os, matplotlib; matplotlib.use('Agg')


import types, re, numpy as np, torch
class _Ent:
    def __init__(s, t): s.text = t
class _Doc:
    def __init__(s, t): s.ents = [_Ent(w) for w in t.split() if w[:1].isupper()]
class _NLP:
    def pipe(s, ts, batch_size=1): return [_Doc(t) for t in ts]
spacy = types.SimpleNamespace(load=lambda n: _NLP())


import os, glob, json, re, time, numpy as np, torch, pandas as pd
DEVICE = "cpu"

def find_one(pattern, what):
    hits = sorted(glob.glob(f"./_sandbox/phaseE/tE/in/**/{pattern}", recursive=True))
    assert hits, f"{what}: '{pattern}' not found. Add the input notebook (see the table at the top)"
    if len(hits) > 1:
        assert len({os.path.getsize(h) for h in hits}) == 1, f"{pattern}: different copies found: {hits}"
    return hits[0]
def l2n(X): return X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)

OUT = "./_sandbox/phaseE/tE/work/phaseE5"; os.makedirs(OUT, exist_ok=True)
N = 200
texts = json.load(open(find_one("texts.json", "phaseA")))["victim_p_L32"]
Y_true = np.load(find_one("gtr_victim_L32.npy", "phaseA")).astype(np.float32)
eval_idx = np.random.default_rng(0).choice(5000, 1024, replace=False); pick = eval_idx[:N]
truth = [texts[i] for i in pick]
mean_B = np.load(find_one("mean_B.npy", "PhaseC2_FP32"))
norm_B = json.load(open(find_one("results_C_fp32.json", "PhaseC2_FP32")))["norm_B"]
D = json.load(open(find_one("results_D.json", "PhaseD_INVERSION")))
VICTIM = {"fp32": "bge-large_victim_L32.npy", "pq128x8": "decoded_victim_pq128x8_bge-large_L32.npy",
          "svd256": "decoded_victim_svd256_bge-large_L32.npy"}
EXPECT = {
    "picked seeds": "reproduce Phase D (fp32 0.238, pq128x8 0.175, svd256 0.251 token F1) within 0.005",
    "collapsed seeds (top-1 < 0.05)": "token F1 near the floor (0.10-0.13)",
    "good seeds": "token F1 within about 0.02 of each other inside a scheme",
    "svd256 vs fp32": "unknown; decided by the rule in the header",
}
for k, v in EXPECT.items(): print(f"- {k}: {v}")


import sacrebleu
class _V2T:
    class models:
        class model_utils:
            @staticmethod
            def mean_pool(h, m): return (h * m[..., None]).sum(1) / m.sum(1, keepdim=True)
    @staticmethod
    def invert_embeddings(embeddings, corrector, num_steps, sequence_beam_width=0):
        return ["Paris war king" for _ in range(len(embeddings))]
vec2text = _V2T; corrector = None
class _BE(dict):
    def to(s, d): return s
gtr_tok = lambda ts, **k: _BE({"input_ids": torch.zeros(len(ts), 4, dtype=torch.long), "attention_mask": torch.ones(len(ts), 4)})
class _Enc:
    def __call__(s, input_ids, attention_mask): return types.SimpleNamespace(last_hidden_state=torch.randn(len(input_ids), 4, 768))
gtr_enc = _Enc(); nlp = spacy.load("x")


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

prep = lambda X, m: l2n(X - m).astype(np.float32)
rows, F1 = [], {}
for scheme, f in VICTIM.items():
    R = json.load(open(find_one(f"results_C_{scheme}.json", f"PhaseC2_{scheme.upper()}")))
    picked = R["phaseD_pick"][scheme]["seed"]
    mean_A = np.load(find_one(f"mean_A_{scheme}.npy", f"PhaseC2_{scheme.upper()}"))
    EA = prep(np.load(find_one(f, f"victim vectors {scheme}")).astype(np.float32)[pick], mean_A)
    for run in [r for r in R["runs"] if r["scheme"] == scheme]:
        s = run["seed"]; t0 = time.time()
        W = np.load(find_one(f"W_{scheme}_s{s}.npy", f"PhaseC2_{scheme.upper()}"))
        preds = invert((l2n(EA @ W) * norm_B + mean_B).astype(np.float32))
        r, f1 = score(preds, f"{scheme} seed {s}")
        F1[(scheme, s)] = f1
        rows.append({"scheme": scheme, "seed": s, "picked": s == picked, "phaseC_top1": run["refine2"]["top1"],
                     "self_consistency": run["self_consistency"], "token_f1": r["token_f1"], "bleu": r["bleu"],
                     "entity_recovery": r["entity_recovery"], "gtr_cos": r["gtr_cos_recovered_vs_true"],
                     "minutes": (time.time() - t0) / 60})
        if s == picked:
            assert abs(r["token_f1"] - D["schemes"][scheme]["token_f1"]) < 0.005, \
                f"{scheme} picked seed does not reproduce Phase D: stop and tell Aman"
        pd.DataFrame(rows).to_csv(f"{OUT}/per_seed.csv", index=False)
T = pd.DataFrame(rows); print(T.round(4).to_string(index=False))

summary = {}
for scheme in VICTIM:
    t = T[T.scheme == scheme]; g = t[t.phaseC_top1 >= 0.5]
    summary[scheme] = {"all_seeds_mean_f1": float(t.token_f1.mean()), "all_seeds_sd": float(t.token_f1.std(ddof=1)),
                       "good_seeds": int(len(g)), "good_seeds_mean_f1": float(g.token_f1.mean()),
                       "good_seeds_sd": float(g.token_f1.std(ddof=1)) if len(g) > 1 else 0.0,
                       "good_seeds_mean_entities": float(g.entity_recovery.mean())}
gap = summary["svd256"]["good_seeds_mean_f1"] - summary["fp32"]["good_seeds_mean_f1"]
thr = max(summary["svd256"]["good_seeds_sd"], summary["fp32"]["good_seeds_sd"])
verdict = "svd256 leaks more than fp32" if gap > thr else "no reliable difference between svd256 and fp32"
print(json.dumps(summary, indent=1)); print(f"svd256 - fp32 (good seeds) = {gap:+.4f}; threshold {thr:.4f} -> {verdict}")
json.dump({"expected": EXPECT, "per_seed": rows, "summary": summary, "svd_vs_fp32": {"gap": gap, "threshold": thr, "verdict": verdict}},
          open(f"{OUT}/results_E5.json", "w"), indent=1, default=float)
import matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize=(9, 4.5)); cols = {"fp32": "tab:gray", "pq128x8": "tab:orange", "svd256": "tab:blue"}
for scheme in VICTIM:
    t = T[T.scheme == scheme]
    ax.scatter(t.phaseC_top1, t.token_f1, s=60, color=cols[scheme], label=scheme)
ax.axhline(D["controls"]["floor_random_other_passage"]["token_f1"], color="gray", ls="--", label="random other passage (floor)")
ax.set_xlabel("Translation top-1 of the seed (Phase C2, 8,192 passages)"); ax.set_ylabel("Token F1 of recovered text (1,000 passages)")
ax.set_title("Text leak per seed: good translation seeds leak, collapsed seeds do not"); ax.legend(fontsize=8)
plt.tight_layout(); plt.savefig(f"{OUT}/phaseE5_seeds.png", dpi=140); 
print(sorted(os.listdir(OUT)))