import nbformat as nbf
from pathlib import Path

HERE = Path(__file__).parent
nb = nbf.v4.new_notebook()
cells = []
md = lambda s: cells.append(nbf.v4.new_markdown_cell(s.strip()))
code = lambda s: cells.append(nbf.v4.new_code_cell(s.strip()))

md("""
# Notebook 1: small end-to-end test of the whole pipeline

Neural Nexus, vector-DB compression project.

What this notebook does, in order:

1. Load 18,000 Natural Questions passages, cut each to its first 32 tokens.
2. Victim side: embed 10,000 passages with BGE-large (the victim database) and their 10,000 questions.
3. Attacker side: embed 8,000 different passages with BGE and GTR-base (the attacker's own public data).
4. Build the 6 compressed databases (fp32, int8, int4, PQ, binary, SVD-256) and run correctness tests on each.
5. Retrieval: Top-10 recall for each compressed database (how useful the DB still is).
6. Positive control: vec2text on true GTR vectors. Must give BLEU above 80, otherwise the notebook stops.
7. Alignment: a supervised linear map from BGE space to GTR space (ridge regression). This is an UPPER BOUND, not the unsupervised attack. The unsupervised attack (mini-vec2vec) is Notebook 2.
8. Inversion: vec2text on 100 aligned vectors per scheme, scored with BLEU and token F1.
9. Results table, plot, and files saved to /kaggle/working.

Settings before you press Run All:
- Accelerator: GPU T4 (x1 or x2, only one is used)
- Internet: ON
- Expected runtime: about 25-35 minutes.
""")

code(r"""
# Cell 1 - install. transformers is pinned because vec2text 0.0.13 breaks on newer versions
# (this exact setup worked on a Kaggle T4 on 23 Oct 2026).
!pip install -q "transformers==4.46.3" "vec2text==0.0.13" faiss-cpu sacrebleu
""")

code(r"""
# Cell 2 - settings. Change sizes here only.
import os, json, time, random
import numpy as np, torch

SEED = 0
N_VICTIM = 10_000      # victim database size
N_ATTACKER = 8_000     # attacker's own public passages (disjoint from the victim DB)
N_INVERT = 100         # victim passages we try to reconstruct per scheme
MAX_TOKENS = 32        # the pretrained inverter was trained on 32-token texts
INV_STEPS = 20         # 20 correction steps: BLEU 91.2 in our 23 Oct benchmark
INV_BATCH = 25
OUT = "/kaggle/working/nb1"
os.makedirs(OUT, exist_ok=True)

random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
assert DEVICE == "cuda", "No GPU. Turn on GPU T4 in the notebook settings."
print(torch.__version__, torch.cuda.get_device_name(0))
import transformers; print("transformers", transformers.__version__)
""")

code(r"""
# Cell 3 - data. Cut every passage to its first 32 GTR (T5) tokens,
# drop duplicates, then split into victim DB and attacker set (no overlap).
from datasets import load_dataset
from transformers import AutoTokenizer

gtr_tok = AutoTokenizer.from_pretrained("sentence-transformers/gtr-t5-base")
ds = load_dataset("sentence-transformers/natural-questions", split="train")
print(ds)

order = np.random.default_rng(SEED).permutation(len(ds))
need = N_VICTIM + N_ATTACKER
queries, passages, seen = [], [], set()
for start in range(0, len(order), 4000):
    idx = order[start:start + 4000].tolist()
    rows = ds.select(idx)
    enc = gtr_tok(list(rows["answer"]), max_length=MAX_TOKENS, truncation=True)
    cut = gtr_tok.batch_decode(enc["input_ids"], skip_special_tokens=True)
    for q, p in zip(list(rows["query"]), cut):
        p = p.strip()
        if p and p not in seen:
            seen.add(p); queries.append(q); passages.append(p)
    if len(passages) >= need:
        break
queries, passages = queries[:need], passages[:need]

victim_q, victim_p = queries[:N_VICTIM], passages[:N_VICTIM]
att_p = passages[N_VICTIM:need]
inv_idx = np.random.default_rng(SEED + 1).choice(N_VICTIM, N_INVERT, replace=False)
json.dump({"victim_q": victim_q, "victim_p": victim_p, "att_p": att_p,
           "inv_idx": inv_idx.tolist()}, open(f"{OUT}/texts.json", "w"))
print(len(victim_p), "victim passages,", len(att_p), "attacker passages")
print("example:", victim_p[0])
""")

code(r"""
# Cell 4 - embedding functions.
# BGE-large: CLS token, L2-normalised (how BGE is meant to be used). Questions get BGE's search prefix.
# GTR: mean-pooled T5 ENCODER output with NO dense layer -- the only variant the pretrained
#      inverter accepts (the standard sentence-transformers version gave BLEU 0.3 on 23 Oct).
from transformers import AutoModel
import vec2text

BGE_PREFIX = "Represent this sentence for searching relevant passages: "

@torch.no_grad()
def embed_bge(texts, model, tok, bs=64):
    out = []
    for i in range(0, len(texts), bs):
        b = tok(texts[i:i+bs], padding=True, truncation=True, max_length=512, return_tensors="pt").to(DEVICE)
        h = model(**b).last_hidden_state[:, 0]
        out.append(torch.nn.functional.normalize(h.float(), dim=-1).cpu())
    return torch.cat(out).numpy().astype(np.float32)

@torch.no_grad()
def embed_gtr(texts, encoder, tok, bs=64):
    out = []
    for i in range(0, len(texts), bs):
        b = tok(texts[i:i+bs], return_tensors="pt", max_length=128, truncation=True,
                padding="max_length").to(DEVICE)
        h = encoder(input_ids=b["input_ids"], attention_mask=b["attention_mask"]).last_hidden_state
        out.append(vec2text.models.model_utils.mean_pool(h, b["attention_mask"]).float().cpu())
    return torch.cat(out).numpy().astype(np.float32)

def cached(name, fn):
    path = f"{OUT}/{name}.npy"
    if os.path.exists(path):
        print("loaded", path); return np.load(path)
    t = time.time(); arr = fn(); np.save(path, arr)
    print(f"{name}: {arr.shape} in {time.time()-t:.0f}s"); return arr
""")

code(r"""
# Cell 5 - victim side: BGE embeddings of the victim DB and its questions,
# plus BGE embeddings of the attacker's passages (attacker can run any public model).
bge_tok = AutoTokenizer.from_pretrained("BAAI/bge-large-en-v1.5")
bge = AutoModel.from_pretrained("BAAI/bge-large-en-v1.5", torch_dtype=torch.float16).to(DEVICE).eval()

X_db  = cached("bge_victim_p", lambda: embed_bge(victim_p, bge, bge_tok))
Q_db  = cached("bge_victim_q", lambda: embed_bge([BGE_PREFIX + q for q in victim_q], bge, bge_tok))
X_att = cached("bge_att_p",    lambda: embed_bge(att_p, bge, bge_tok))

del bge; torch.cuda.empty_cache()
""")

code(r"""
# Cell 6 - attacker side: GTR embeddings.
# Y_att: attacker's own passages (used to fit the alignment).
# Y_db : victim passages in GTR space -- ONLY used for scoring, the attacker never sees these.
gtr_enc = AutoModel.from_pretrained("sentence-transformers/gtr-t5-base").encoder.to(DEVICE).eval()
Y_att = cached("gtr_att_p",    lambda: embed_gtr(att_p, gtr_enc, gtr_tok))
Y_db  = cached("gtr_victim_p", lambda: embed_gtr(victim_p, gtr_enc, gtr_tok))
del gtr_enc; torch.cuda.empty_cache()
print("GTR norm (mean):", float(np.linalg.norm(Y_db, axis=1).mean()))
""")

code((HERE / "cell_codecs.py").read_text())
code((HERE / "cell_metrics.py").read_text())

code(r"""
# Cell 9 - build the 6 compressed victim databases and test each one.
# The run stops here if any codec fails a test.
import pandas as pd
codecs = make_codecs()
checks = []
for c in codecs:
    t = time.time()
    c.build(X_db)
    r = roundtrip_check(c, X_db[:2000])
    r["build_s"] = round(time.time() - t, 1)
    checks.append(r)
checks = pd.DataFrame(checks)
checks.to_csv(f"{OUT}/codec_checks.csv", index=False)
checks
""")

code(r"""
# Cell 10 - retrieval: does the right passage come back in the top 10 for its question?
gold = np.arange(N_VICTIM)
retrieval = {c.name: topk_recall(c, Q_db, gold, k=10) * 100 for c in codecs}
pd.Series(retrieval, name="top10_recall_%").round(1)
""")

code(r"""
# Cell 11 - positive control: invert the TRUE GTR vectors of the 100 test passages.
# If this is not above 80 BLEU, something is broken (wrong embedding variant, wrong
# library version) and every later number would be meaningless -- so we stop.
corrector = vec2text.load_pretrained_corrector("gtr-base")
refs = [victim_p[i] for i in inv_idx]

def invert(vectors):
    preds = []
    for i in range(0, len(vectors), INV_BATCH):
        e = torch.tensor(vectors[i:i+INV_BATCH], dtype=torch.float32, device=DEVICE)
        preds += vec2text.invert_embeddings(embeddings=e, corrector=corrector, num_steps=INV_STEPS)
    return preds

t = time.time()
ctrl_preds = invert(Y_db[inv_idx])
ctrl = text_scores(ctrl_preds, refs)
print(f"positive control: {ctrl}  ({time.time()-t:.0f}s)")
assert ctrl["bleu"] > 80, "Positive control failed. Do not trust anything below. Send this output to the team."
""")

code(r"""
# Cell 12 - alignment + inversion for each compressed DB.
# Attacker's view: he stole the compressed index, so he can decode it with its own codebooks,
# and he can push his OWN passages through the same codec. He fits a linear map
# (compressed BGE -> GTR) on his own pairs, then applies it to the victim's stolen vectors.
# This is a SUPERVISED upper bound: a real unsupervised attacker can only do worse or equal.
rows, examples = [], {}
pool = np.arange(1000)   # alignment top-1 is measured against 1,000 victim candidates
for c in codecs:
    t = time.time()
    X_att_c = c.decode(c.encode(X_att))          # attacker's data, compressed the same way
    W, b, lam = fit_oracle(X_att_c, Y_att)
    X_vic_c = c.decode(c.codes)                  # victim's stolen vectors
    Y_hat = X_vic_c @ W + b
    al = alignment_scores(Y_hat[pool], Y_db[pool])
    preds = invert(Y_hat[inv_idx])
    ts = text_scores(preds, refs)
    rows.append({"scheme": c.name, "bytes_per_vector": c.bytes_per_vector(),
                 "top10_recall_%": retrieval[c.name], "ridge_lambda": lam,
                 "align_top1_%": al["align_top1"] * 100, "align_mean_rank": al["align_mean_rank"],
                 "align_mean_cos": al["align_mean_cos"], **ts, "time_s": round(time.time() - t)})
    examples[c.name] = list(zip(refs[:3], preds[:3]))
    print(rows[-1])

res = pd.DataFrame(rows)
res.to_csv(f"{OUT}/results_oracle.csv", index=False)
json.dump({"positive_control": ctrl, "examples": examples, "results": rows},
          open(f"{OUT}/results_oracle.json", "w"), indent=1)
res.round(2)
""")

code(r"""
# Cell 13 - plot: privacy leak (BLEU) and usefulness (Top-10 retrieval) against storage size.
import matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(res["bytes_per_vector"], res["top10_recall_%"], "o", color="tab:blue", label="Top-10 retrieval (%)")
ax.plot(res["bytes_per_vector"], res["bleu"], "s", color="tab:red", label="Reconstruction BLEU (oracle alignment)")
for _, r in res.iterrows():
    ax.annotate(r["scheme"], (r["bytes_per_vector"], r["bleu"]), textcoords="offset points", xytext=(5, 5), fontsize=9)
ax.axhline(ctrl["bleu"], color="gray", ls="--", lw=1, label=f"Positive control BLEU ({ctrl['bleu']:.1f})")
ax.set_xscale("log")
ax.set_xlabel("Storage per vector (bytes, log scale)")
ax.set_ylabel("Score (0-100)")
ax.set_title("Notebook 1: retrieval vs reconstruction per compression scheme\n(BGE-large victim, 10k NQ passages, supervised linear alignment)")
ax.legend(fontsize=9); ax.grid(alpha=0.3)
plt.tight_layout(); plt.savefig(f"{OUT}/nb1_tradeoff.png", dpi=150); plt.show()
""")

code(r"""
# Cell 14 - read three reconstructions per scheme with your own eyes.
for name, ex in examples.items():
    print(f"=== {name}")
    for ref, pred in ex:
        print("  TRUE:", ref); print("  GOT :", pred); print()
print("Files saved in", OUT, ":", os.listdir(OUT))
""")

md("""
## What to send back to the team after it finishes

- The table from Cell 12 (also saved as `nb1/results_oracle.csv`).
- The plot `nb1/nb1_tradeoff.png`.
- The positive-control BLEU from Cell 11.

How to read it:
- If an int8 or int4 BLEU is close to the positive control, those compressions do not protect anything, even against a linear map.
- If PQ or binary BLEU is near zero even here, then the unsupervised attack (Notebook 2) will also fail on them, because it cannot beat this supervised upper bound.
- To keep the files, click Save Version (Save & Run All) so /kaggle/working is stored with the notebook.
""")

nb["cells"] = cells
nb["metadata"] = {"kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"},
                  "language_info": {"name": "python"}}
nbf.write(nb, HERE / "nb1_small_end_to_end.ipynb")
print("written", len(cells), "cells")
