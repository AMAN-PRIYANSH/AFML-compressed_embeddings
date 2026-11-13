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

OUT = "./_sandbox/phaseE/tE/work/phaseE1"; os.makedirs(OUT, exist_ok=True)
L, N = 32, 200
texts = json.load(open(find_one("texts.json", "phaseA")))["victim_p_L32"]
Y_true = np.load(find_one("gtr_victim_L32.npy", "phaseA")).astype(np.float32)
X_vic = np.load(find_one("bge-large_victim_L32.npy", "phaseA")).astype(np.float32)
X_att = np.load(find_one("bge-large_att_L32.npy", "phaseA")).astype(np.float32)
Y_att = np.load(find_one("gtr_att_L32.npy", "phaseA")).astype(np.float32)
keep = np.load(find_one("attacker_keep_mask.npy", "PhaseC2_FP32"))
mean_A = np.load(find_one("mean_A_fp32.npy", "PhaseC2_FP32")); mean_B = np.load(find_one("mean_B.npy", "PhaseC2_FP32"))
C2 = json.load(open(find_one("results_C_fp32.json", "PhaseC2_FP32"))); norm_B = C2["norm_B"]
D = json.load(open(find_one("results_D.json", "PhaseD_INVERSION")))
assert len(keep) == len(X_att) == len(Y_att) and keep.sum() == 1450
X_att, Y_att = X_att[keep], Y_att[keep]                 # same attacker passages as Phase C2 (near-copies removed)
eval_idx = np.random.default_rng(0).choice(5000, 1024, replace=False); pick = eval_idx[:N]
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
for epoch in range(2):          # same recipe as Phase C v1
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

import matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize=(8, 4.5))
names = ["mini-vec2vec\n(no pairs)"] + list(results["maps"])
vals = [D["schemes"]["fp32"]["token_f1"]] + [results["maps"][k]["inversion"]["token_f1"] for k in results["maps"]]
ax.bar(names, vals, color=["tab:red"] + ["tab:orange"] * 3 + ["tab:gray"], label="token F1 of recovered text")
ax.axhline(D["controls"]["positive_true_vectors"]["token_f1"], color="tab:green", ls="--", label="true vectors (upper bound)")
ax.axhline(D["controls"]["floor_random_other_passage"]["token_f1"], color="gray", ls=":", label="random other passage (floor)")
ax.set_xlabel("How the stolen fp32 vector was translated into GTR space"); ax.set_ylabel("Token F1 vs the real passage")
ax.set_title("Text recovered: unsupervised vs supervised translation"); ax.legend(fontsize=8); ax.set_ylim(0, 1)
plt.tight_layout(); plt.savefig(f"{OUT}/phaseE1_supervised.png", dpi=140); 
json.dump(results, open(f"{OUT}/results_E1.json", "w"), indent=1)
print(sorted(os.listdir(OUT)))