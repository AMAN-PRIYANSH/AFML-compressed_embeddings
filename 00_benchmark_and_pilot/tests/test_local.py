import numpy as np, time
exec(open('cell_codecs.py').read()); exec(open('cell_metrics.py').read())
rng=np.random.default_rng(0)
# synthetic "embeddings" with structure (low-rank + noise), unit norm like BGE
Z=rng.normal(size=(18000,64)).astype(np.float32)
X=Z@rng.normal(size=(64,1024)).astype(np.float32)+0.3*rng.normal(size=(18000,1024)).astype(np.float32)
X/=np.linalg.norm(X,axis=1,keepdims=True)
Y=Z@rng.normal(size=(64,768)).astype(np.float32)+0.1*rng.normal(size=(18000,768)).astype(np.float32)
Xdb,Xatt=X[:10000],X[10000:]; Ydb,Yatt=Y[:10000],Y[10000:]
Q=Xdb+0.05*rng.normal(size=Xdb.shape).astype(np.float32); Q/=np.linalg.norm(Q,axis=1,keepdims=True)
gold=np.arange(10000)
for c in make_codecs():
    t=time.time(); c.build(Xdb)
    rt=roundtrip_check(c,Xdb[:2000])
    r10=topk_recall(c,Q[:2000],gold[:2000])
    W,b,lam=fit_oracle(c.decode(c.encode(Xatt)),Yatt)
    Yh=c.decode(c.codes[:1000])@W+b
    al=alignment_scores(Yh,Ydb[:1000])
    print(rt, 'top10',round(r10,3),'lam',lam,al, f'{time.time()-t:.1f}s')
print(text_scores(["the cat sat on the mat","hello world"],["the cat sat on a mat","hello world"]))
print(token_f1("a b c","a b d"))
