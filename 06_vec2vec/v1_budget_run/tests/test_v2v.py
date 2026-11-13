import numpy as np, sys, os, json, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from v2v_runner import run
rng = np.random.default_rng(0)
dz=32
Z = rng.normal(size=(9000, dz))
PA, PB = rng.normal(size=(dz,1024)), rng.normal(size=(dz,768))
XA = Z[:3000]@PA + 0.1*rng.normal(size=(3000,1024))
YB = Z[3000:7000]@PB + 0.1*rng.normal(size=(4000,768))
XE = Z[7000:]@PA; YE = Z[7000:]@PB
t=time.time()
best, logp = run(os.path.abspath('../ext/vec2vec'), XA, YB, XE, YE, os.path.abspath('test_out'), budget_hours=0.03,
                 val_size=1024, extra_args={"mixed_precision": "no", "val_bs": 512, "top_k_size": 512, "top_k_batches": 2})
print("BEST", best, f"{time.time()-t:.0f}s")
print(open(logp).read()[-1500:])
