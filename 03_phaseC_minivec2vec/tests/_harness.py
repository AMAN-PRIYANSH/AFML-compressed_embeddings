import nbformat, os
nb = nbformat.read("phaseC2_minivec2vec_samesource.ipynb", 4)
W = os.path.abspath("tC/working"); I = os.path.abspath("tC/input"); H = os.path.abspath("tC/hf")
out = ["import os, matplotlib; matplotlib.use('Agg')"]
for c in nb.cells:
    if c.cell_type != "code": continue
    s = c.source.replace("/kaggle/working", W).replace("/kaggle/input", I)
    s = s.replace('SCHEMES = ["fp32"]', 'SCHEMES = ["fp32", "pq128x8"]')
    s = s.replace("from huggingface_hub import hf_hub_download", f"hf_hub_download = lambda repo, f: os.path.join({H!r}, f)")
    s = s.replace(">= 0.85, \"CONTROL", ">= 0.0, \"CONTROL")
    s = s.replace("50000", "5000").replace("200000", "20000").replace('"bge-large", 8192', '"bge-large", 1024')
    s = s.replace("len(A) - 8192", "len(A) - 1024")
    s = s.replace("def mini_vec2vec(XA, XB, seed=0, n_runs=30,", "def mini_vec2vec(XA, XB, seed=0, n_runs=3,")
    s = s.replace("refine1_iters=100,", "refine1_iters=10,").replace("refine2_clusters=500,", "refine2_clusters=40,")
    s = s.replace("< 0.005, \"attacker set", "< 1.1, \"attacker set"); s = s.replace("len(Y_att) > 15000", "len(Y_att) > 1500"); s = s.replace("for epoch in range(30)", "for epoch in range(3)").replace("plt.show()", "")
    out.append(s)
open("tC_nb.py", "w").write("\n\n".join(out)); print("harness ok")
