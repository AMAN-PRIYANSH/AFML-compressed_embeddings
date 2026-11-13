import numpy as np, json, torch
r=np.random.default_rng(3); dz=48; k=50
C=r.normal(size=(k,8))@r.normal(size=(8,dz))*1.5; w=r.dirichlet(np.ones(k)*.5)
def S(n, noise=2.0): return C[r.choice(k,size=n,p=w)]+noise*r.normal(size=(n,dz))
def O(d): Q,_=np.linalg.qr(r.normal(size=(d,dz))); return Q.T
PA,PB,PC=O(1024),O(768),O(768)
Zv,Za,Zs=S(5000),S(20000),S(3000)
Zs[:50] = Zv[:50] + 0.01*r.normal(size=(50,dz))          # 50 planted near-copies
f=lambda X: X.astype(np.float32)
np.save("tC/input/phaseA/bge-large_victim_L32.npy", f(Zv@PA)); np.save("tC/input/phaseA/gtr_victim_L32.npy", f(Zv@PB))
np.save("tC/input/phaseA/gtr_att_L32.npy", f(Zs@PB)); np.save("tC/input/phaseA/bge-large_att_L32.npy", f(Zs@PA))
json.dump({"phase":"A_fp32_embeddings"}, open("tC/input/phaseA/manifest.json","w"))
json.dump({"victim_p_L32":[f"victim text {i}" for i in range(5000)], "att_p_L32":[f"attacker text {i}" for i in range(3000)]}, open("tC/input/phaseA/texts.json","w"))
np.save("tC/input/phaseA2b/gtr_attextra_L32.npy", f(Za@PB))
np.save("tC/input/phaseB_pq/decoded_victim_pq128x8_bge-large_L32.npy", f(Zv@PA + 0.3*r.normal(size=(5000,1024))/32))
Zc=S(12000,0.6); torch.save(torch.tensor(f(Zc@O(768))), "tC/hf/e5.pt"); torch.save(torch.tensor(f(Zc@PC)), "tC/hf/gtr.pt")
print("fake ok")
