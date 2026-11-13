import nbformat as nbf
from pathlib import Path

HERE = Path(__file__).parent
cells = []
md = lambda s: cells.append(nbf.v4.new_markdown_cell(s.strip()))
code = lambda s: cells.append(nbf.v4.new_code_cell(s.strip()))

md("""
# Phase A2: a bigger attacker dataset (GTR only)

Neural Nexus, vector-DB compression project. Run once, by Aman.

Why: Phase A had only 69,569 unique NQ passages, so the attacker set came out at 19,569 instead of 40,000.
vec2vec needs about 50,000+ unaligned vectors per side to work at all, and mini-vec2vec used about 26,000.
So the attacker gets 200,000 extra passages of the same kind (Wikipedia passages from the NQ DPR corpus,
the exact corpus vec2vec and mini-vec2vec used), embedded with GTR at 32 and 128 tokens.
Any passage that also appears in the victim database is removed.

Expected time: about 30-45 minutes on GPU T4 x2 (this notebook uses BOTH T4s, one half of the data each).

How to run:
1. Right panel: Add Input -> Your Work -> Notebook Output Files -> `phaseA_fp32_embeddings` (or the `nn-vecdb-fp32` dataset).
2. Settings: Accelerator = **GPU T4 x2**, Internet = On.
3. Save Version -> Save & Run All (Commit).
4. When done: Output tab -> New Dataset -> name `nn-vecdb-att-extra`, share with the team.
""")

code(r"""
# Cell 1 - install (same pins as Phase A, so the GTR vectors are computed identically)
!pip install -q "transformers==4.46.3" "vec2text==0.0.13"
""")

code(r"""
# Cell 2 - settings and Phase A input
import os, glob, json, time, hashlib, datetime, threading
import numpy as np, torch

SEED = 1                 # different from Phase A's seed on purpose
N_EXTRA = 200_000
LENGTHS = (32, 128)
OUT = "/kaggle/working"

found = [p for p in glob.glob("/kaggle/input/**/manifest.json", recursive=True)
         if json.load(open(p)).get("phase") == "A_fp32_embeddings"]
assert len(found) == 1, f"Add the Phase A output as input (found {found})"
IN = os.path.dirname(found[0])
manA = json.load(open(found[0]))
assert manA["checks"]["positive_control_bleu_L32"] > 80
texts_A = json.load(open(f"{IN}/texts.json"))

n_gpu = torch.cuda.device_count()
assert n_gpu >= 1, "No GPU: set Accelerator = GPU T4 x2"
print("GPUs:", n_gpu, [torch.cuda.get_device_name(i) for i in range(n_gpu)])
import transformers; print(torch.__version__, transformers.__version__)
""")

code(r"""
# Cell 3 - texts: NQ DPR corpus (dev split, 850k passages), random 200k after removing
# anything whose first 32 tokens match a Phase A victim or attacker passage.
from datasets import load_dataset
from transformers import AutoTokenizer

gtr_tok = AutoTokenizer.from_pretrained("sentence-transformers/gtr-t5-base")
ds = load_dataset("jxm/nq_corpus_dpr", split="dev")
print(ds)

def cut(texts, L, bs=4000):
    out = []
    for i in range(0, len(texts), bs):
        ids = gtr_tok(texts[i:i+bs], max_length=L, truncation=True)["input_ids"]
        out += [t.strip() for t in gtr_tok.batch_decode(ids, skip_special_tokens=True)]
    return out

order = np.random.default_rng(SEED).permutation(len(ds))[: int(N_EXTRA * 1.5)]
raw = list(ds.select(order.tolist())["text"])
p32 = cut(raw, 32)
banned = set(texts_A["victim_p_L32"]) | set(texts_A["att_p_L32"])
keep, seen, n_overlap = [], set(), 0
for i, t in enumerate(p32):
    if not t or t in seen:
        continue
    if t in banned:
        n_overlap += 1; continue
    seen.add(t); keep.append(i)
    if len(keep) == N_EXTRA:
        break
assert len(keep) == N_EXTRA, f"only {len(keep)} usable passages"
print(f"removed {n_overlap} passages that were in the victim/attacker sets of Phase A")

texts = {"source": "jxm/nq_corpus_dpr dev", "dpr_row": [int(order[i]) for i in keep],
         "extra_p_L32": [p32[i] for i in keep]}
texts["extra_p_L128"] = cut([raw[i] for i in keep], 128)
json.dump(texts, open(f"{OUT}/texts_extra.json", "w"))
print("example:", texts["extra_p_L32"][0])
""")

code(r"""
# Cell 4 - GTR embedding on both GPUs. Same function as Phase A (mean-pooled T5 encoder,
# no dense layer, max_length=128 padding) so these vectors live in exactly the same space.
from transformers import AutoModel
import vec2text

@torch.no_grad()
def embed_gtr(texts, encoder, device, tok, bs=128):
    out = []
    for i in range(0, len(texts), bs):
        b = tok(texts[i:i+bs], return_tensors="pt", max_length=128, truncation=True,
                    padding="max_length").to(device)
        h = encoder(input_ids=b["input_ids"], attention_mask=b["attention_mask"]).last_hidden_state
        out.append(vec2text.models.model_utils.mean_pool(h, b["attention_mask"]).float().cpu())
    return torch.cat(out).numpy().astype(np.float32)

devices = [f"cuda:{i}" for i in range(n_gpu)]
encoders = {d: AutoModel.from_pretrained("sentence-transformers/gtr-t5-base").encoder.to(d).eval() for d in devices}
# one tokenizer per thread: a fast tokenizer shared by two threads can crash ("Already borrowed")
toks = {d: AutoTokenizer.from_pretrained("sentence-transformers/gtr-t5-base") for d in devices}

def embed_parallel(texts):
    parts = np.array_split(np.arange(len(texts)), len(devices))
    results = [None] * len(devices)
    def work(k):
        results[k] = embed_gtr([texts[i] for i in parts[k]], encoders[devices[k]], devices[k], toks[devices[k]])
    threads = [threading.Thread(target=work, args=(k,)) for k in range(len(devices))]
    for t in threads: t.start()
    for t in threads: t.join()
    return np.concatenate(results)

E = {}
for L in LENGTHS:
    path = f"{OUT}/gtr_attextra_L{L}.npy"
    if os.path.exists(path):
        E[L] = np.load(path); continue
    t0 = time.time()
    E[L] = embed_parallel(texts[f"extra_p_L{L}"])
    np.save(path, E[L])
    print(f"L{L}: {E[L].shape} in {time.time()-t0:.0f}s")
""")

code(r"""
# Cell 5 - checks. Any failure stops the notebook.
checks = {}
for L in LENGTHS:
    X = E[L]
    assert X.shape == (N_EXTRA, 768) and np.isfinite(X).all()
    # (a) same space as Phase A: compare norms with Phase A's attacker GTR vectors
    ref = np.load(f"{IN}/gtr_att_L{L}.npy", mmap_mode="r")[:20000]
    n_new, n_ref = np.linalg.norm(X, axis=1), np.linalg.norm(ref, axis=1)
    checks[f"norm_mean_new_vs_phaseA_L{L}"] = [float(n_new.mean()), float(n_ref.mean())]
    assert abs(n_new.mean() / n_ref.mean() - 1) < 0.05, "norms differ: wrong GTR variant?"
    # (b) the same vector computed twice must match: re-embed 64 texts on GPU 0
    again = embed_gtr(texts[f"extra_p_L{L}"][:64], encoders[devices[0]], devices[0], toks[devices[0]])
    err = float(np.abs(again - X[:64]).max())
    checks[f"reembed_max_abs_diff_L{L}"] = err
    assert err < 1e-3, err
    # (c) no hidden copies of victim passages: closest victim vector for 5,000 extra vectors
    vic = np.load(f"{IN}/gtr_victim_L{L}.npy")
    vn = vic / np.linalg.norm(vic, axis=1, keepdims=True)
    idx = np.random.default_rng(0).choice(N_EXTRA, 5000, replace=False)
    xn = X[idx] / np.linalg.norm(X[idx], axis=1, keepdims=True)
    mx = np.concatenate([(xn[i:i+500] @ vn.T).max(1) for i in range(0, 5000, 500)])
    checks[f"frac_near_duplicate_of_victim_L{L}"] = float((mx > 0.99).mean())
    print(f"L{L}: near-duplicates of victim passages: {(mx > 0.99).mean()*100:.2f}%")
    assert (mx > 0.99).mean() < 0.01
print(json.dumps(checks, indent=1))
print("ALL CHECKS PASSED")
""")

code(r"""
# Cell 6 - manifest
def sha(path, chunk=1 << 24):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while b := f.read(chunk): h.update(b)
    return h.hexdigest()
files = sorted(f for f in os.listdir(OUT) if f.endswith(".npy"))
manifest = {"project": "Neural Nexus vector-DB compression", "phase": "A2_attacker_extra",
            "created_utc": datetime.datetime.utcnow().isoformat(timespec="seconds"),
            "seed": SEED, "n_extra": N_EXTRA, "lengths": list(LENGTHS), "source": texts["source"],
            "phaseA_manifest_created": manA["created_utc"], "n_removed_overlap": n_overlap,
            "versions": {"torch": torch.__version__, "transformers": transformers.__version__},
            "files": {f: {"shape": list(np.load(f"{OUT}/{f}", mmap_mode="r").shape), "sha256": sha(f"{OUT}/{f}")} for f in files},
            "checks": checks}
json.dump(manifest, open(f"{OUT}/manifest_A2.json", "w"), indent=1)
print(json.dumps({k: v for k, v in manifest.items() if k != "files"}, indent=1))
""")

nb = nbf.v4.new_notebook()
nb["cells"] = cells
nb["metadata"] = {"kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"},
                  "language_info": {"name": "python"}}
nbf.write(nb, HERE / "phaseA2_attacker_extra.ipynb")
print("written", len(cells), "cells")
