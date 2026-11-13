# sanity: victim and attacker hold DIFFERENT samples of the same clustered distribution, hidden rotation
import numpy as np, time, sys
sys.path.insert(0, ".")
from minivec2vec import *
rng = np.random.default_rng(1); dz = 64; k = 60
centers = rng.normal(size=(k, 8)) @ rng.normal(size=(8, dz)) * 1.5          # topics with varied distances
w = rng.dirichlet(np.ones(k) * 0.5)                                           # uneven topic sizes
def sample(n): c = rng.choice(k, size=n, p=w); return centers[c] + 0.6 * rng.normal(size=(n, dz))
def orth(d): Q, _ = np.linalg.qr(rng.normal(size=(d, dz))); return Q.T
PA, PB = orth(1024), orth(768)
ZA, ZB, ZE = sample(12000), sample(30000), sample(2000)
XA, XB, EA, EB = ZA @ PA, ZB @ PB, ZE @ PA, ZE @ PB
mA, mB = XA.mean(0), XB.mean(0)
XA, XB, EA, EB = prep(XA, mA), prep(XB, mB), prep(EA, mA), prep(EB, mB)
print("oracle", align_eval(EA, EB, (PA.T @ PB).astype(np.float32)))
t = time.time()
W, h = mini_vec2vec(XA, XB, seed=0, eval_fn=lambda W: align_eval(EA, EB, W), refine2_clusters=200)
print(f"{time.time()-t:.0f}s", h)
