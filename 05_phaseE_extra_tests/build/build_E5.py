import sys
sys.path.insert(0, ".")
from build_E import write, M, C, FIND, METRICS, INV   # noqa: E402

e5 = [M("""
# PhaseE5_SEEDS: does the text leak depend on the seed? (all 5 seeds, inverted)

Neural Nexus, vector-DB compression project (team leader: Aman).

| | |
|---|---|
| **Time** | about 1.5-2 hours (15 inversions of 1,000 passages) |
| **Settings** | Accelerator = **GPU T4** (x2 is fine), Internet = **On** |
| **Inputs** | `phaseA_fp32_embeddings`, `PHASE_B_pq128x8` (amanpriyansh/phase-b-pq128x8), `PhaseB_SVD256` (pravithmohandas/phaseb-svd256), `PhaseC2_FP32`, `PhaseC2_PQ128X8`, `PhaseC2_SVD256`, `PhaseD_INVERSION` |
| **Nothing to change** | |
| **Output** | `phaseE5/`: results_E5.json, per-seed table, plot |

Why: Phase D and E0 used ONE seed per scheme (the one an attacker picks). E0 found SVD256 leaking slightly MORE
than fp32, but that could be seed luck. Here every one of the 5 seeds of fp32, pq128x8 and svd256 is inverted on
the same 1,000 passages, with the same settings as Phase D.
Check built in: the picked seed must reproduce Phase D's token F1 (within 0.005), else it stops.
Rule fixed before running: "SVD256 leaks more than fp32" is kept only if svd256's mean token F1 over its seeds with
top-1 >= 0.5 is above fp32's mean over its seeds with top-1 >= 0.5 by more than the larger of the two seed spreads
(standard deviations).
"""),
C(r"""
!pip install -q "transformers==4.46.3" "vec2text==0.0.13" sacrebleu spacy
!python -m spacy download en_core_web_sm -q
"""),
C(r"""
import os, glob, json, re, time, numpy as np, torch, pandas as pd
DEVICE = "cuda"; assert torch.cuda.is_available(), "Set Accelerator = GPU T4"
""" + FIND + r"""
OUT = "/kaggle/working/phaseE5"; os.makedirs(OUT, exist_ok=True)
N = 1000
texts = json.load(open(find_one("texts.json", "phaseA")))["victim_p_L32"]
Y_true = np.load(find_one("gtr_victim_L32.npy", "phaseA")).astype(np.float32)
eval_idx = np.random.default_rng(0).choice(50000, 8192, replace=False); pick = eval_idx[:N]
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
"""),
C(INV),
C(r"""
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
"""),
C(r"""
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
plt.tight_layout(); plt.savefig(f"{OUT}/phaseE5_seeds.png", dpi=140); plt.show()
print(sorted(os.listdir(OUT)))
""")]
write(e5, "PhaseE5_SEEDS.ipynb")
