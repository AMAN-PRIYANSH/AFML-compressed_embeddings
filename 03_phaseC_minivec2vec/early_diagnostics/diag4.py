import numpy as np
exec(open('minivec2vec.py').read())
rng = np.random.default_rng(1); k, dz = 30, 12
centers = rng.normal(size=(k, dz)) * 2.5
def sample(n): c = rng.integers(k, size=n); return centers[c] + 0.8*rng.normal(size=(n, dz))
def orth(d): Q, _ = np.linalg.qr(rng.normal(size=(d, dz))); return Q.T
PA, PB = orth(1024), orth(768)
mk = lambda Z, P: (Z @ P + 0.01*rng.normal(size=(len(Z), P.shape[1]))).astype(np.float32)
ZA, ZB, ZE = sample(12000), sample(12000), sample(2000)
XAr, XBr, EAr, EBr = mk(ZA,PA), mk(ZA[::-1].copy(),PB), mk(ZE,PA), mk(ZE,PB)
mA, mB = XAr.mean(0), XBr.mean(0)
XA, XB, EA, EB = prep(XAr,mA), prep(XBr,mB), prep(EAr,mA), prep(EBr,mB)
print('oracle', align_eval(EA, EB, (PA.T@PB).astype(np.float32)))
W, h = mini_vec2vec(XA, XB, n_runs=10, refine1_iters=50, refine2_clusters=200, eval_fn=lambda W: align_eval(EA,EB,W), log=print)
