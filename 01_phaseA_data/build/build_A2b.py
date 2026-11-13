import nbformat as nbf
from pathlib import Path

HERE = Path(__file__).parent
cells = []
md = lambda s: cells.append(nbf.v4.new_markdown_cell(s.strip()))
code = lambda s: cells.append(nbf.v4.new_code_cell(s.strip()))

md("""
# Phase A2b: verify the A2 attacker vectors (no re-embedding)

Neural Nexus, vector-DB compression project. Run once, by Aman.

Phase A2 finished both embedding steps (200,000 x 768 at 32 and 128 tokens) and then stopped at one check:
"mean vector length must be within 5% of Phase A's". That 5% rule was a guess, and a wrong one:
A2's passages are cut from the middle of Wikipedia articles (DPR chunks), Phase A's start at paragraph starts,
so their average vector length can differ for a real reason. It does not show the vectors are wrong.

This notebook checks what actually matters, directly:
1. Re-embed 256 random A2 passages now. They must match the saved vectors (proves same GTR variant, rows in the right order).
2. Re-embed 256 Phase A attacker passages. They must match Phase A's saved vectors (proves today's setup equals Phase A's).
3. Near-copies of victim passages inside the attacker set must be under 1%.
4. Measure (not judge) how different the A2 passages are from Phase A's, so Phase C can test whether it matters.

Then it copies the verified A2 files to its own output, with a manifest, so teammates add ONE input for A2.

Expected time: about 5-10 minutes. Settings: Accelerator = GPU T4 (x1 is enough), Internet = On.
Inputs (Add Input -> Your Work): notebook output of `phaseA_fp32_embeddings` AND of `phaseA2_attacker_extra`.
""")

code(r"""
# Cell 1 - install (same pins as Phase A and A2)
!pip install -q "transformers==4.46.3" "vec2text==0.0.13"
""")

code(r"""
# Cell 2 - find both inputs
import os, glob, json, shutil, hashlib, datetime
import numpy as np, torch

OUT = "/kaggle/working"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
assert DEVICE == "cuda", "Set Accelerator = GPU T4"

manA_path = [p for p in glob.glob("/kaggle/input/**/manifest.json", recursive=True)
             if json.load(open(p)).get("phase") == "A_fp32_embeddings"]
a2_texts = glob.glob("/kaggle/input/**/texts_extra.json", recursive=True)
assert len(manA_path) == 1, f"Add phaseA_fp32_embeddings output as input ({manA_path})"
assert len(a2_texts) == 1, f"Add phaseA2_attacker_extra output as input ({a2_texts})"
IN_A, IN_A2 = os.path.dirname(manA_path[0]), os.path.dirname(a2_texts[0])
manA = json.load(open(manA_path[0]))
texts_A = json.load(open(f"{IN_A}/texts.json"))
texts_X = json.load(open(a2_texts[0]))
LENGTHS = (32, 128)
X = {L: np.load(f"{IN_A2}/gtr_attextra_L{L}.npy") for L in LENGTHS}
print(IN_A, IN_A2, {L: X[L].shape for L in LENGTHS})
""")

code(r"""
# Cell 3 - what we expect BEFORE looking (written down so the run cannot move the goalposts)
EXPECT = {
    "shape": "(200000, 768) for both lengths, no NaN",
    "reembed_A2_max_abs_diff": "below 1e-3 (same GPU type, same code: usually about 1e-5)",
    "reembed_phaseA_max_abs_diff": "below 1e-3 (shows this session reproduces Phase A exactly)",
    "frac_near_copy_of_victim": "below 1% (727 exact text copies were already removed in A2)",
    "norm_ratio_A2_over_phaseA": "anything; measured only. DPR mid-article chunks vs paragraph starts",
    "nn_mixing_A2_share": "about 50% if A2 looks like Phase A's passages, lower if they differ (measured only)",
}
for k, v in EXPECT.items(): print(f"{k:32s} {v}")
""")

code(r"""
# Cell 4 - the checks
from transformers import AutoTokenizer, AutoModel
import vec2text

tok = AutoTokenizer.from_pretrained("sentence-transformers/gtr-t5-base")
enc = AutoModel.from_pretrained("sentence-transformers/gtr-t5-base").encoder.to(DEVICE).eval()

@torch.no_grad()
def embed_gtr(texts, bs=128):   # identical to Phase A / A2
    out = []
    for i in range(0, len(texts), bs):
        b = tok(texts[i:i+bs], return_tensors="pt", max_length=128, truncation=True, padding="max_length").to(DEVICE)
        h = enc(input_ids=b["input_ids"], attention_mask=b["attention_mask"]).last_hidden_state
        out.append(vec2text.models.model_utils.mean_pool(h, b["attention_mask"]).float().cpu())
    return torch.cat(out).numpy().astype(np.float32)

def unit(M): return M / np.linalg.norm(M, axis=1, keepdims=True)
rng = np.random.default_rng(0)
checks = {}
for L in LENGTHS:
    V = X[L]
    assert V.shape == (200000, 768) and np.isfinite(V).all()
    # 1. A2 rows match their texts
    i = rng.choice(len(V), 256, replace=False)
    d1 = float(np.abs(embed_gtr([texts_X[f"extra_p_L{L}"][j] for j in i]) - V[i]).max())
    # 2. this session reproduces Phase A
    PA = np.load(f"{IN_A}/gtr_att_L{L}.npy")
    k = rng.choice(len(PA), 256, replace=False)
    d2 = float(np.abs(embed_gtr([texts_A[f"att_p_L{L}"][j] for j in k]) - PA[k]).max())
    # 3. near-copies of victim passages
    vic = unit(np.load(f"{IN_A}/gtr_victim_L{L}.npy"))
    s = rng.choice(len(V), 5000, replace=False)
    xs = unit(V[s])
    mx = np.concatenate([(xs[a:a+500] @ vic.T).max(1) for a in range(0, 5000, 500)])
    near = float((mx > 0.99).mean())
    # 4. how different A2 is from Phase A (measured only)
    ratio = float(np.linalg.norm(V, axis=1).mean() / np.linalg.norm(PA, axis=1).mean())
    pa_a, pa_b = PA[:2000], PA[2000:4000]
    pool = unit(np.concatenate([pa_b, V[rng.choice(len(V), 2000, replace=False)]]))
    nn = np.argmax(unit(pa_a) @ pool.T, axis=1)
    mixing = float((nn >= 2000).mean())          # share of nearest neighbours that come from A2
    checks[f"L{L}"] = {"reembed_A2_max_abs_diff": d1, "reembed_phaseA_max_abs_diff": d2,
                       "frac_near_copy_of_victim": near, "norm_ratio_A2_over_phaseA": ratio,
                       "nn_mixing_A2_share": mixing}
    print(f"L{L}:", json.dumps(checks[f"L{L}"], indent=1))
    assert d1 < 1e-3, "A2 vectors do not match their texts"
    assert d2 < 1e-3, "this session does not reproduce Phase A"
    assert near < 0.01, "too many near-copies of victim passages in the attacker set"
print("ALL CHECKS PASSED")
""")

code(r"""
# Cell 5 - copy the verified A2 files to this notebook's output, with a manifest
def sha(path, chunk=1 << 24):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while b := f.read(chunk): h.update(b)
    return h.hexdigest()
for f in ["texts_extra.json"] + [f"gtr_attextra_L{L}.npy" for L in LENGTHS]:
    shutil.copy(f"{IN_A2}/{f}", f"{OUT}/{f}")
import transformers
manifest = {"project": "Neural Nexus vector-DB compression", "phase": "A2_attacker_extra",
            "created_utc": datetime.datetime.utcnow().isoformat(timespec="seconds"),
            "source": texts_X["source"], "n_extra": 200000, "lengths": list(LENGTHS),
            "phaseA_manifest_created": manA["created_utc"],
            "versions": {"torch": torch.__version__, "transformers": transformers.__version__},
            "files": {f: {"shape": list(np.load(f"{OUT}/{f}", mmap_mode="r").shape), "sha256": sha(f"{OUT}/{f}")}
                      for f in sorted(os.listdir(OUT)) if f.endswith(".npy")},
            "expected": EXPECT, "checks": checks}
json.dump(manifest, open(f"{OUT}/manifest_A2.json", "w"), indent=1)
print(json.dumps(checks, indent=1)); print(sorted(os.listdir(OUT)))
""")

nb = nbf.v4.new_notebook()
nb["cells"] = cells
nb["metadata"] = {"kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"},
                  "language_info": {"name": "python"}}
nbf.write(nb, HERE / "phaseA2b_verify_attacker_extra.ipynb")
print("written", len(cells), "cells")
