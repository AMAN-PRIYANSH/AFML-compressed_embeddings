import nbformat as nbf
from pathlib import Path

HERE = Path(__file__).parent
cells = []
md = lambda s: cells.append(nbf.v4.new_markdown_cell(s.strip()))
code = lambda s: cells.append(nbf.v4.new_code_cell(s.strip()))

md("""
# Phase B: build ONE compressed database and measure retrieval

Neural Nexus, vector-DB compression project. One teammate runs this once per compression scheme.

The only thing you change is `SCHEME` in Cell 2:
`fp32`, `int8`, `int4`, `pq128x8`, `binary`, or `svd256`.

What it does, for every victim model and text length in the Phase A dataset:
1. Builds the compressed database from the fp32 victim vectors.
2. Runs correctness tests (stored size, decode really inverts encode, search agrees with a manual check).
3. Measures retrieval: does the right passage come back for its question (top-1, top-10, top-100)?
4. Saves the stored codes, the decoded vectors (what an attacker gets after stealing the index), and the compressor itself.

How to run:
1. Right panel: **Add Input** -> **Notebooks** -> search `phaseA_fp32_embeddings` (owner Aman Priyansh) -> click +. Add ONLY this one input.
2. Settings: Accelerator = **None** (CPU is enough, saves GPU hours), Internet = **On**.
3. Set `SCHEME` in Cell 2. Rename the notebook to `phaseB-<scheme>` (example `phaseB-int8`).
4. Save Version -> Save & Run All (Commit). About 10-30 minutes (pq128x8 can take up to about 1 hour: training the codebooks is slow on CPU).
5. When done: Share the notebook with Aman (Share -> add username `amanpriyansh`, Can view). Also download the `results_*.json` file from the Output tab and send it on the group.
""")

code(r"""
# Cell 1 - install
!pip install -q faiss-cpu
""")

code(r"""
# Cell 2 - settings. CHANGE ONLY SCHEME.
SCHEME = "int8"   # one of: fp32, int8, int4, pq128x8, binary, svd256

import os, glob, json, time, hashlib, datetime
import numpy as np
OUT = "/kaggle/working"
N_QUERY_EVAL = 10_000   # questions used for retrieval scoring
SEED = 0

found = glob.glob("/kaggle/input/**/manifest.json", recursive=True)
assert len(found) == 1, f"Add the phaseA_fp32_embeddings notebook output as the only input (found {found})"
IN = os.path.dirname(found[0])
manifest = json.load(open(found[0]))
assert manifest["phase"] == "A_fp32_embeddings", "wrong input dataset"
assert all(manifest["checks"][k] > 0.2 for k in manifest["checks"] if k.startswith("top10_fp32")), "Phase A checks failed"
print("input:", IN)
print("Phase A positive control BLEU:", manifest["checks"]["positive_control_bleu_L32"])
""")

code((HERE / "codecs.py").read_text())

code(r"""
# Cell 4 - which (victim model, length) combinations exist in the Phase A dataset
combos = []
for f in sorted(manifest["files"]):
    if f.startswith("bge") and "_victim_L" in f:
        model_name, rest = f[:-4].split("_victim_L")
        combos.append((model_name, int(rest)))
print(combos)

def make_codec(name):
    return {c.name: c for c in make_codecs()}[name]
assert SCHEME in {c.name for c in make_codecs()}, f"unknown SCHEME {SCHEME}"
""")

code(r"""
# Cell 5 - build, test, measure, save. One block per (model, length).
# Expected decoded-vs-original cosine, from theory and our synthetic tests:
EXPECT_COS = {"fp32": (0.9999, 1.0001), "int8": (0.999, 1.0), "int4": (0.97, 1.0),
              "pq128x8": (0.80, 0.99), "binary": (0.60, 0.95), "svd256": (None, None)}
# Expected top-10 retrieval, written BEFORE the run. fp32 values were measured in Phase A
# (5,000 questions); the other rows are expected drops relative to fp32, from how much each
# scheme distorts the vectors.
FP32_TOP10 = {("bge-large", 32): 0.859, ("bge-large", 128): 0.955, ("bge-base", 32): 0.850}
EXPECT_DROP = {"fp32": (0.00, 0.02), "int8": (0.00, 0.01), "int4": (0.00, 0.04),
               "pq128x8": (0.02, 0.15), "binary": (0.03, 0.20), "svd256": (0.01, 0.10)}
print("Expected for", SCHEME)
print("  decoded-vs-original cosine:", EXPECT_COS[SCHEME])
for (m, L), v in FP32_TOP10.items():
    lo, hi = EXPECT_DROP[SCHEME]
    print(f"  top-10 retrieval {m} L{L}: about {v-hi:.3f} to {v-lo:.3f}")
import faiss
rng = np.random.default_rng(SEED)
results = {"scheme": SCHEME, "created_utc": datetime.datetime.utcnow().isoformat(timespec="seconds"),
           "input_manifest_created": manifest["created_utc"], "faiss": faiss.__version__, "runs": []}

for model_name, L in combos:
    t0 = time.time()
    tag = f"{SCHEME}_{model_name}_L{L}"
    X_vic = np.load(f"{IN}/{model_name}_victim_L{L}.npy")
    X_att = np.load(f"{IN}/{model_name}_att_L{L}.npy")
    Q = np.load(f"{IN}/{model_name}_query.npy")

    codec = make_codec(SCHEME).build(X_vic)
    chk = roundtrip_check(codec, X_vic[:5000])

    lo, hi = EXPECT_COS[SCHEME]
    cos = chk["mean_cos_to_original"]
    chk["cos_in_expected_range"] = None if lo is None else bool(lo <= cos <= hi)
    if SCHEME == "svd256":
        Xr = codec.encode(X_vic[:5000]) @ codec.V.T
        chk["energy_kept"] = float((np.linalg.norm(Xr) / np.linalg.norm(X_vic[:5000])) ** 2)

    # retrieval: question i should find passage i
    qi = rng.choice(len(Q), min(N_QUERY_EVAL, len(Q)), replace=False)
    I = codec.search(Q[qi], 100)
    hit = (I == qi[:, None])
    ret = {f"top{k}": float(hit[:, :k].any(1).mean()) for k in (1, 10, 100)}

    # save: stored codes, decoded vectors (victim and attacker), and the compressor
    np.save(f"{OUT}/codes_{tag}.npy", codec.codes)
    np.save(f"{OUT}/decoded_victim_{tag}.npy", codec.decode(codec.codes).astype(np.float32))
    np.save(f"{OUT}/decoded_att_{tag}.npy", codec.decode(codec.encode(X_att)).astype(np.float32))
    if isinstance(codec, FaissCodec):
        faiss.write_index(codec.index, f"{OUT}/index_{tag}.faiss")
    elif SCHEME == "svd256":
        np.save(f"{OUT}/svd_basis_{tag}.npy", codec.V)

    lo_d, hi_d = EXPECT_DROP[SCHEME]
    ref = FP32_TOP10[(model_name, L)]
    chk["top10_in_expected_range"] = bool(ref - hi_d - 0.02 <= ret["top10"] <= ref - lo_d + 0.02)
    run = {"model": model_name, "length": L, **chk, **{f"retrieval_{k}": v for k, v in ret.items()},
           "n_queries": int(len(qi)), "seconds": round(time.time() - t0, 1)}
    results["runs"].append(run)
    print(json.dumps(run, indent=1))

json.dump(results, open(f"{OUT}/results_{SCHEME}.json", "w"), indent=1)
""")

code(r"""
# Cell 6 - summary table. Send this (or results_<scheme>.json) to the group.
import pandas as pd
df = pd.DataFrame(results["runs"])
cols = ["model", "length", "bytes_per_vector", "mean_cos_to_original", "cos_in_expected_range",
        "retrieval_top1", "retrieval_top10", "top10_in_expected_range", "retrieval_top100", "search_agrees", "seconds"]
print(f"SCHEME = {SCHEME}")
df[[c for c in cols if c in df.columns]].round(4)
""")

md("""
## Reading the table

- `cos_in_expected_range` and `top10_in_expected_range` should be True. If either is False, do not fix anything: send the table to Aman. (False is not automatically a bug: it can be a real result. Aman and Claude decide.)
- `retrieval_top10` for `fp32` is the best possible. Every other scheme is compared against it.
- `search_agrees` should be about 1.0. It checks the compressed index against a manual calculation.
""")

nb = nbf.v4.new_notebook()
nb["cells"] = cells
nb["metadata"] = {"kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"},
                  "language_info": {"name": "python"}}
nbf.write(nb, HERE / "phaseB_compress_one_scheme.ipynb")
print("written", len(cells), "cells")
