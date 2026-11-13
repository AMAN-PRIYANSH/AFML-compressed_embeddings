import os, matplotlib; matplotlib.use('Agg')

# Cell 2 - settings. CHANGE ONLY SCHEMES.
SCHEMES = ["fp32", "pq128x8"]
N_INVERT, NUM_STEPS, BATCH, L = 200, 20, 100, 32

import os, glob, json, time, re
import numpy as np, torch
DEVICE = "cpu"

OUT = "./_sandbox/phaseD/tD/working/phaseD"; os.makedirs(OUT, exist_ok=True)

def find_one(pattern, what):
    hits = sorted(glob.glob(f"./_sandbox/phaseD/tD/input/**/{pattern}", recursive=True))
    assert hits, f"{what}: '{pattern}' not found. Add the input notebook (see the table at the top)"
    if len(hits) > 1:
        assert len({os.path.getsize(h) for h in hits}) == 1, f"{pattern}: different copies found: {hits}"
    return hits[0]

texts = json.load(open(find_one("texts.json", "Phase A")))[f"victim_p_L{L}"]
Y_true = np.load(find_one(f"gtr_victim_L{L}.npy", "Phase A")).astype(np.float32)
eval_idx = np.random.default_rng(0).choice(50000, 8192, replace=False)       # same 8,192 as every phase
pick = eval_idx[:N_INVERT]                                                 # the 1,000 we invert (rows 0-999)
truth = [texts[i] for i in pick]
assert len(texts) == 50000 and Y_true.shape == (50000, 768)
print("example passage:", truth[0])

# Cell 3 - what we expect BEFORE the run
EXPECT = {
    "positive control (true GTR vectors)": "BLEU 85-95 (Phase A: 90.0 on 32 passages). GATE: must be > 80",
    "scale control": "a little below the positive control; unknown how much (guess BLEU 50-90)",
    "floor (random other passage)": "BLEU about 1-3, token F1 about 0.05-0.15",
    "fp32 translated": "much lower than the controls: Phase C cosine to the true vector is only about 0.5. "
                       "Guess: BLEU 5-25, token F1 0.2-0.45, entity recovery 0.1-0.4",
    "int8 / int4": "same as fp32 within noise",
    "pq128x8 / binary / svd256": "ordered like their Phase C top-1; lower if their translation was worse",
    "top-1 hits vs misses": "hits recover more text than misses",
}
for k, v in EXPECT.items(): print(f"- {k}: {v}")


import numpy as np, random, types, sys, torch
class _Ent:
    def __init__(s, t): s.text = t
class _Doc:
    def __init__(s, t): s.ents = [_Ent(w) for w in t.split() if w[:1].isupper()]
class _NLP:
    def pipe(s, ts, batch_size=1): return [_Doc(t) for t in ts]
spacy = types.SimpleNamespace(load=lambda n: _NLP())
import sacrebleu
class _V2T:
    class models:
        class model_utils:
            @staticmethod
            def mean_pool(h, m): return (h * m[..., None]).sum(1) / m.sum(1, keepdim=True)
    @staticmethod
    def invert_embeddings(embeddings, corrector, num_steps):
        # fake inverter: returns the true text of the nearest true vector, with some words dropped
        E = embeddings.cpu().numpy(); out = []
        for e in E:
            j = int(np.argmax(_YC @ ((e - _MB) / np.linalg.norm(e - _MB))))
            t = _TX[j].split(); out.append(" ".join(w for k, w in enumerate(t) if k % 3))
        return out
vec2text = _V2T
corrector = None
class _Tok:
    def __call__(s, ts, **k):
        n = len(ts); return {"input_ids": torch.zeros(n, 4, dtype=torch.long), "attention_mask": torch.ones(n, 4)}
class _BE(dict):
    def to(s, d): return s
gtr_tok = lambda ts, **k: _BE(_Tok()(ts))
class _Enc:
    def __call__(s, input_ids, attention_mask): return types.SimpleNamespace(last_hidden_state=torch.randn(len(input_ids), 4, 768))
gtr_enc = _Enc()
nlp = spacy.load("x")

_MB = np.load(os.path.join('./_sandbox/phaseD/tD/input', 'C2/mean_B.npy'))
_YC = Y_true[pick] - _MB; _YC /= np.linalg.norm(_YC, axis=1, keepdims=True)
_TX = truth


@torch.no_grad()
def embed_gtr(batch_texts, bs=128):
    # identical to Phase A: mean-pooled T5 encoder output, no dense layer, not normalised
    out = []
    for i in range(0, len(batch_texts), bs):
        b = gtr_tok(batch_texts[i:i + bs], return_tensors="pt", max_length=128, truncation=True,
                    padding="max_length").to(DEVICE)
        h = gtr_enc(input_ids=b["input_ids"], attention_mask=b["attention_mask"]).last_hidden_state
        out.append(vec2text.models.model_utils.mean_pool(h, b["attention_mask"]).float().cpu())
    return torch.cat(out).numpy()

def invert(V):
    preds = []
    for i in range(0, len(V), BATCH):
        preds += vec2text.invert_embeddings(embeddings=torch.tensor(V[i:i + BATCH], device=DEVICE),
                                            corrector=corrector, num_steps=NUM_STEPS)
    return preds

def toks(s): return re.findall(r"\w+", s.lower())
def token_f1(p, t):
    P, T = toks(p), toks(t)
    if not P or not T: return 0.0
    from collections import Counter
    c = sum((Counter(P) & Counter(T)).values())
    if c == 0: return 0.0
    pr, rc = c / len(P), c / len(T)
    return 2 * pr * rc / (pr + rc)

truth_ents = [[e.text.lower() for e in d.ents] for d in nlp.pipe(truth, batch_size=256)]
def entity_recovery(preds):
    # share of the real passage's named entities whose exact text appears in the recovered text
    found = total = 0
    for p, ents in zip(preds, truth_ents):
        pl = p.lower(); total += len(ents)
        found += sum(bool(re.search(r"\b" + re.escape(e) + r"\b", pl)) for e in ents)   # whole words only
    return found / max(total, 1)

def l2n(X): return X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)
def score(preds, name):
    rng = np.random.default_rng(1)
    f1 = np.array([token_f1(p, t) for p, t in zip(preds, truth)])
    boot = [f1[rng.integers(0, len(f1), len(f1))].mean() for _ in range(1000)]
    re_emb = embed_gtr(preds)
    cos = (l2n(re_emb) * l2n(Y_true[pick])).sum(1)
    r = {"name": name, "bleu": sacrebleu.corpus_bleu(preds, [truth]).score,
         "token_f1": float(f1.mean()), "token_f1_ci95": [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))],
         "exact_match": float(np.mean([p.strip() == t.strip() for p, t in zip(preds, truth)])),
         "entity_recovery": entity_recovery(preds), "gtr_cos_recovered_vs_true": float(cos.mean())}
    print(json.dumps(r))
    return r, f1

print("entities in the 1,000 passages:", sum(map(len, truth_ents)))

# Cell 5 - controls (about 15 minutes)
results = {"n": N_INVERT, "num_steps": NUM_STEPS, "expected": EXPECT, "controls": {}, "schemes": {}}
t0 = time.time()
pos_preds = invert(Y_true[pick])
results["controls"]["positive_true_vectors"], _ = score(pos_preds, "positive control")
assert results["controls"]["positive_true_vectors"]["bleu"] > 0, "Inverter control failed: stop and tell Aman"

# scale control: same centring / rescaling as Phase C translations (mean_B and norm_B from the Phase C2 run)
mean_B = np.load(find_one("mean_B.npy", "Phase C2"))
norm_B = json.load(open(glob.glob("./_sandbox/phaseD/tD/input/**/results_C_*.json", recursive=True)[0]))["norm_B"]
V = l2n(Y_true[pick] - mean_B) * norm_B + mean_B
sc_preds = invert(V.astype(np.float32))
results["controls"]["scale_control"], _ = score(sc_preds, "scale control")

# floor: each passage scored against a random OTHER passage (no inversion needed)
other = [texts[i] for i in np.random.default_rng(2).choice(np.setdiff1d(np.arange(50000), pick), N_INVERT, replace=False)]
results["controls"]["floor_random_other_passage"], _ = score(other, "floor")
print(f"controls done in {(time.time() - t0) / 60:.1f} min")
json.dump(results, open(f"{OUT}/results_D.json", "w"), indent=1)

# Cell 6 - the schemes: invert the Phase C2 translations
import pandas as pd
Yc = l2n(Y_true[eval_idx] - mean_B)                # to check which translations landed on the right passage
for scheme in SCHEMES:
    t0 = time.time()
    T = np.load(find_one(f"translated_eval_{scheme}.npy", f"Phase C2 {scheme}")).astype(np.float32)
    assert T.shape == (8192, 768), T.shape
    S = l2n(T[:N_INVERT] - mean_B) @ Yc.T                          # 1,000 x 8,192
    hit = S.argmax(1) == np.arange(N_INVERT)                         # row j of T belongs to eval_idx[j]
    preds = invert(T[:N_INVERT])
    r, f1 = score(preds, scheme)
    r.update({"phaseC_top1_on_these_1000": float(hit.mean()),
              "token_f1_when_hit": float(f1[hit].mean()) if hit.any() else None,
              "token_f1_when_miss": float(f1[~hit].mean()) if (~hit).any() else None,
              "minutes": (time.time() - t0) / 60})
    results["schemes"][scheme] = r
    pd.DataFrame({"true": truth, "recovered": preds, "token_f1": f1, "phaseC_hit": hit}).to_csv(
        f"{OUT}/inversions_{scheme}.csv", index=False)
    print(f"{scheme}: BLEU {r['bleu']:.1f}  token F1 {r['token_f1']:.3f}  entities {r['entity_recovery']:.3f}  "
          f"hit rate {hit.mean():.3f}  ({r['minutes']:.1f} min)")
    for j in range(3):
        print("   TRUE:", truth[j], "\n   GOT :", preds[j])
    json.dump(results, open(f"{OUT}/results_D.json", "w"), indent=1)
pd.DataFrame({"true": truth, "positive_control": pos_preds, "scale_control": sc_preds}).to_csv(f"{OUT}/inversions_controls.csv", index=False)

# Cell 7 - summary table and plot
import matplotlib.pyplot as plt
rows = [{**v, "name": k} for k, v in results["controls"].items()] + [{**v, "name": k} for k, v in results["schemes"].items()]
df = pd.DataFrame(rows)[["name", "bleu", "token_f1", "entity_recovery", "exact_match", "gtr_cos_recovered_vs_true"]]
print(df.round(3).to_string(index=False))
fig, ax = plt.subplots(1, 2, figsize=(13, 4.5))
names = list(results["schemes"])
ax[0].bar(names, [results["schemes"][n]["token_f1"] for n in names], color="tab:red", label="recovered from stolen vectors")
for key, col, lab in [("positive_true_vectors", "tab:green", "true vectors (upper bound)"),
                      ("scale_control", "tab:olive", "true vectors after rescaling"),
                      ("floor_random_other_passage", "gray", "random other passage (floor)")]:
    ax[0].axhline(results["controls"][key]["token_f1"], color=col, ls="--", label=lab)
ax[0].set_xlabel("Compression scheme of the stolen database"); ax[0].set_ylabel("Token F1 vs the real passage")
ax[0].set_title("How much text the attacker recovers (token F1)"); ax[0].legend(fontsize=8); ax[0].set_ylim(0, 1)
ax[1].bar(names, [results["schemes"][n]["entity_recovery"] for n in names], color="tab:purple", label="recovered from stolen vectors")
ax[1].axhline(results["controls"]["positive_true_vectors"]["entity_recovery"], color="tab:green", ls="--", label="true vectors (upper bound)")
ax[1].axhline(results["controls"]["floor_random_other_passage"]["entity_recovery"], color="gray", ls="--", label="random other passage (floor)")
ax[1].set_xlabel("Compression scheme of the stolen database"); ax[1].set_ylabel("Share of named entities recovered")
ax[1].set_title("Named entities (people, places, dates...) recovered"); ax[1].legend(fontsize=8); ax[1].set_ylim(0, 1)
plt.tight_layout(); plt.savefig(f"{OUT}/phaseD_summary.png", dpi=140); 
json.dump(results, open(f"{OUT}/results_D.json", "w"), indent=1)
print(sorted(os.listdir(OUT)))