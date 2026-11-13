import nbformat, numpy as np, json, os, sys, types
R=os.path.abspath('fk3'); rng=np.random.default_rng(0)
NV, NA, NX = 3000, 4500, 200000
vic = {L: rng.normal(size=(NV,768)).astype(np.float32) for L in (32,128)}
att = {L: rng.normal(size=(NA,768)).astype(np.float32) for L in (32,128)}
ext = {L: (rng.normal(size=(NX,768))*0.9).astype(np.float32) for L in (32,128)}
tA = {'victim_q':['q']*NV}
for L in (32,128):
    tA[f'victim_p_L{L}']=[f'v{L}_{i}' for i in range(NV)]; tA[f'att_p_L{L}']=[f'a{L}_{i}' for i in range(NA)]
    np.save(f'{R}/inA/gtr_victim_L{L}.npy', vic[L]); np.save(f'{R}/inA/gtr_att_L{L}.npy', att[L]); np.save(f'{R}/inA2/gtr_attextra_L{L}.npy', ext[L])
tX={'source':'fake', **{f'extra_p_L{L}':[f'x{L}_{i}' for i in range(NX)] for L in (32,128)}}
json.dump(tA, open(f'{R}/inA/texts.json','w')); json.dump(tX, open(f'{R}/inA2/texts_extra.json','w'))
json.dump({'phase':'A_fp32_embeddings','created_utc':'x'}, open(f'{R}/inA/manifest.json','w'))
lookup={}
for L in (32,128):
    for i,t in enumerate(tA[f'att_p_L{L}']): lookup[t]=att[L][i]
    for i,t in enumerate(tX[f'extra_p_L{L}']): lookup[t]=ext[L][i]
nb=nbformat.read('phaseA2b_verify_attacker_extra.ipynb',4)
cs=[c.source.replace('/kaggle/input',R).replace('/kaggle/working',f'{R}/out') for c in nb.cells if c.cell_type=='code'][1:]
fake_torch=types.SimpleNamespace(cuda=types.SimpleNamespace(is_available=lambda:True), __version__='mock', Tensor=type('T',(),{}))
sys.modules['torch']=fake_torch
g={}
exec(cs[0],g); exec(cs[1],g)
c4=cs[2]; c4=c4[c4.index('def unit'):]
g['embed_gtr']=lambda texts: np.stack([lookup[t] for t in texts])
exec(c4,g)
sys.modules['transformers']=types.SimpleNamespace(__version__='mock')
exec(cs[3],g)
print(sorted(os.listdir(f'{R}/out')))
