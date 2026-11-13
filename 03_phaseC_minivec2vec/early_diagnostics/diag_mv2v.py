import numpy as np, sys
exec(open('minivec2vec.py').read())
def run(hier):
    rng = np.random.default_rng(1); k, dz = 40, 64
    if hier:   # varied distances between topics: centres on a low-dim structure
        centers = rng.normal(size=(k, 6)) @ rng.normal(size=(6, dz)) * 1.0
    else:
        centers = rng.normal(size=(k, dz)) * 2.0
    def sample(n): c = rng.integers(k, size=n); return centers[c] + rng.normal(size=(n, dz))
    def orth(d): Q, _ = np.linalg.qr(rng.normal(size=(d, dz))); return Q.T
    PA, PB = orth(1024), orth(768)
    mk = lambda Z, P: (Z @ P + 0.02*rng.normal(size=(len(Z), P.shape[1]))).astype(np.float32)
    ZA, ZB, ZE = sample(12000), sample(12000), sample(2000)
    XAr, XBr, EAr, EBr = mk(ZA,PA), mk(ZB,PB), mk(ZE,PA), mk(ZE,PB)
    mA, mB = XAr.mean(0), XBr.mean(0)
    XA, XB, EA, EB = prep(XAr,mA), prep(XBr,mB), prep(EAr,mA), prep(EBr,mB)
    Wstar = (PA.T @ PB).astype(np.float32)
    print('hier', hier, 'oracle W* eval', align_eval(EA, EB, Wstar))
    r = np.random.default_rng(0); np.random.seed(0)
    ca, cb = aligned_centroids(XA, XB, r)
    c = (l2n(ca @ Wstar) * l2n(cb)).sum(1)
    print('  one run: cos(centroid_A mapped by W*, matched centroid_B) mean', c.mean().round(3), 'frac>0.9', (c>0.9).mean())
    W, h = mini_vec2vec(XA, XB, n_runs=10, refine1_iters=50, refine2_clusters=200, eval_fn=lambda W: align_eval(EA,EB,W), log=lambda *a: None)
    print('  ', h)
run(False); run(True)
