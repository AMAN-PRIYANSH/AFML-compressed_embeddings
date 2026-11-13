import numpy as np, time
exec(open('minivec2vec.py').read())
rng = np.random.default_rng(1)
k, dz = 40, 64
centers = rng.normal(size=(k, dz)) * 2.0
def sample(n):
    c = rng.integers(k, size=n)
    return centers[c] + rng.normal(size=(n, dz))
def orth(d):
    Q, _ = np.linalg.qr(rng.normal(size=(d, dz))); return Q.T   # dz x d, orthonormal rows
PA, PB = orth(1024), orth(768)
ZA, ZB, ZE = sample(12000), sample(12000), sample(2000)        # A and B never share sentences
mk = lambda Z, P, s: (Z @ P + s * rng.normal(size=(len(Z), P.shape[1]))).astype(np.float32)
XA_raw, XB_raw = mk(ZA, PA, 0.02), mk(ZB, PB, 0.02)
EA_raw, EB_raw = mk(ZE, PA, 0.02), mk(ZE, PB, 0.02)             # eval pairs (same latent)
mA, mB = XA_raw.mean(0), XB_raw.mean(0)
XA, XB, EA, EB = prep(XA_raw, mA), prep(XB_raw, mB), prep(EA_raw, mA), prep(EB_raw, mB)
print("naive (random W) top1:", align_eval(EA, EB, procrustes(rng.normal(size=(50,1024)), rng.normal(size=(50,768))))["top1"])
t = time.time()
W, hist = mini_vec2vec(XA, XB, seed=0, n_runs=10, refine1_iters=50, refine2_clusters=200,
                       eval_fn=lambda W: align_eval(EA, EB, W))
print(f"done in {time.time()-t:.0f}s", hist)
print("W shape", W.shape, "column orthonormality error", float(np.abs(W.T @ W - np.eye(768)).max()))
assert hist["refine2"]["top1"] > 0.8
print("MINI-VEC2VEC SYNTHETIC TEST PASSED")
