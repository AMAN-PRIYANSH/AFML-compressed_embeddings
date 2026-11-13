import nbformat, os
nb = nbformat.read("phaseD_inversion.ipynb", 4)
W, I = os.path.abspath("tD/working"), os.path.abspath("tD/input")
mock = r'''
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
'''
out = ["import os, matplotlib; matplotlib.use('Agg')"]
for c in nb.cells:
    if c.cell_type != "code": continue
    s = c.source.replace("/kaggle/working", W).replace("/kaggle/input", I)
    if s.startswith("# Cell 1"): continue
    s = s.replace('DEVICE = "cuda"', 'DEVICE = "cpu"').replace('assert torch.cuda.is_available(), "Set Accelerator = GPU T4"', "")
    s = s.replace('SCHEMES = ["fp32", "int8", "int4", "pq128x8", "binary", "svd256"]', 'SCHEMES = ["fp32", "pq128x8"]')
    s = s.replace("N_INVERT, NUM_STEPS, BATCH, L = 1000,", "N_INVERT, NUM_STEPS, BATCH, L = 200,")
    if s.startswith("# Cell 4"):
        head, rest = s.split("nlp = spacy.load(\"en_core_web_sm\")", 1)
        s = mock + "\n_MB = np.load(os.path.join(%r, 'C2/mean_B.npy'))\n_YC = Y_true[pick] - _MB; _YC /= np.linalg.norm(_YC, axis=1, keepdims=True)\n_TX = truth\n" % I + rest
    s = s.replace("> 80, \"Inverter", "> 0, \"Inverter").replace("plt.show()", "")
    out.append(s)
open("tD_nb.py", "w").write("\n\n".join(out)); print("harness ok")
