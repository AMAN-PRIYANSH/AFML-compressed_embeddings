import nbformat, os, sys
I = os.path.abspath("tE/in"); W = os.path.abspath("tE/work")
mock = r'''
import types, re, numpy as np, torch
class _Ent:
    def __init__(s, t): s.text = t
class _Doc:
    def __init__(s, t): s.ents = [_Ent(w) for w in t.split() if w[:1].isupper()]
class _NLP:
    def pipe(s, ts, batch_size=1): return [_Doc(t) for t in ts]
spacy = types.SimpleNamespace(load=lambda n: _NLP())
'''
mock_inv = r'''
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
'''
for nbname in sys.argv[1:]:
    nb = nbformat.read(nbname, 4); out = ["import os, matplotlib; matplotlib.use('Agg')", mock]
    for c in nb.cells:
        if c.cell_type != "code" or c.source.startswith("!pip"): continue
        if c.source.startswith("import subprocess, sys"): continue
        s = c.source.replace("/kaggle/working", W).replace("/kaggle/input", I)
        s = s.replace('DEVICE = "cuda"; assert torch.cuda.is_available(), "Set Accelerator = GPU T4"', 'DEVICE = "cpu"')
        s = s.replace("choice(50000,", "choice(5000,").replace("np.arange(50000)", "np.arange(5000)").replace(", 8192,", ", 1024,")
        s = s.replace("keep.sum() == 18854", "keep.sum() == 1450").replace("L, N = 32, 1000", "L, N = 32, 200")
        s = s.replace("import sacrebleu, spacy\n", "import sacrebleu\n").replace("for epoch in range(30)", "for epoch in range(2)")
        s = s.replace("> 75, \"Inverter", "> -1, \"Inverter").replace("plt.show()", "")
        if "import vec2text, sacrebleu, spacy" in s:
            head, rest = s.split("nlp = spacy.load(\"en_core_web_sm\")", 1); s = mock_inv + rest
        s = s.replace("N, K, ROUNDS, LAM = 1000,", "N, K, ROUNDS, LAM = 200,")
        s = s.replace("N = 1000\n", "N = 200\n")
        s = s.replace("T.shape == (8192, 768)", "T.shape == (1024, 768)")
        out.append(s)
    open("t_" + nbname.replace(".ipynb", ".py"), "w").write("\n\n".join(out))
print("ok")
