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

OUT = "./_sandbox/phaseE/tE/work/phaseE2"; os.makedirs(OUT, exist_ok=True)
N = 200
texts = json.load(open(find_one("texts.json", "phaseA")))["victim_p_L32"]
Y_true = np.load(find_one("gtr_victim_L32.npy", "phaseA")).astype(np.float32)
eval_idx = np.random.default_rng(0).choice(5000, 1024, replace=False); pick = eval_idx[:N]
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
            assert r["bleu"] > -1, "Inverter control failed: stop and tell Aman"
        json.dump(results, open(f"{OUT}/results_E2.json", "w"), indent=1)
pd.DataFrame(csv).to_csv(f"{OUT}/inversions_decoding.csv", index=False)
df = pd.DataFrame(results["runs"]).pivot(index="set", columns="config", values="token_f1")
print(df.round(3))

import matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize=(9, 4.5)); w = 0.25
sets = list(SETS); x = np.arange(len(sets))
for i, (cfg, *_ ) in enumerate(CONFIGS):
    ax.bar(x + (i - 1) * w, [df.loc[s, cfg] for s in sets], w, label=cfg)
ax.set_xticks(x); ax.set_xticklabels(sets, rotation=10)
ax.set_xlabel("Vectors given to the inverter"); ax.set_ylabel("Token F1 vs the real passage (200 passages)")
ax.set_title("Does a stronger inverter recover more text?"); ax.legend(); ax.set_ylim(0, 1)
plt.tight_layout(); plt.savefig(f"{OUT}/phaseE2_decoding.png", dpi=140); 
print(sorted(os.listdir(OUT)))