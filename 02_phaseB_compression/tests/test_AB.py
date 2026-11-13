import nbformat, numpy as np, json, sys, types, time, os
ROOT = os.path.abspath('fk2')
W, INP = f'{ROOT}/working', f'{ROOT}/input'
os.makedirs(W, exist_ok=True); os.makedirs(f'{INP}/nn-vecdb-fp32', exist_ok=True)
fix = lambda s: s.replace('/kaggle/working', W).replace('/kaggle/input', INP)
A = nbformat.read('phaseA_fp32_embeddings.ipynb', 4)
codecells = [c.source for c in A.cells if c.cell_type == 'code']
g = {}
for m in ['torch','transformers','vec2text']: sys.modules[m] = types.SimpleNamespace(__version__='mock', Tensor=type('T',(),{}))
import torch, transformers, vec2text
g.update(dict(np=np, json=json, os=os, time=time, torch=torch, transformers=transformers, vec2text=vec2text,
              hashlib=__import__('hashlib'), datetime=__import__('datetime')))
NV, NA = 6000, 5000
g.update(SEED=0, N_VICTIM=NV, N_ATTACKER=NA, LENGTHS=(32,128), OUT=W, BGE_PREFIX='p: ')
rng = np.random.default_rng(0)
def unit(X): return (X/np.linalg.norm(X,axis=1,keepdims=True)).astype(np.float32)
E = {}
Zv = rng.normal(size=(NV,48)); Za = rng.normal(size=(NA,48))
for m, d in [('bge-large',1024),('bge-base',768)]:
    P = rng.normal(size=(48,d))
    E[f'{m}_query'] = unit(Zv@P + 0.8*rng.normal(size=(NV,d)))
    for L in ((32,128) if m=='bge-large' else (32,)):
        E[f'{m}_victim_L{L}'] = unit(Zv@P + 0.5*rng.normal(size=(NV,d)))
        E[f'{m}_att_L{L}'] = unit(Za@P + 0.5*rng.normal(size=(NA,d)))
Pg = rng.normal(size=(48,768))
for L in (32,128):
    E[f'gtr_victim_L{L}'] = (Zv@Pg*0.05 + 0.02*rng.normal(size=(NV,768))).astype(np.float32)
    E[f'gtr_att_L{L}'] = (Za@Pg*0.05 + 0.02*rng.normal(size=(NA,768))).astype(np.float32)
for k,v in E.items(): np.save(f'{W}/{k}.npy', v)
g['E'] = E
cell7 = codecells[6]; cell7 = cell7[:cell7.index('# (e)')]
exec(fix(cell7), g)
g['checks']['positive_control_bleu_L32'] = 91.0   # mocked (needs GPU + vec2text)
exec(fix(codecells[7]), g)
print('PHASE A checks+manifest OK')
for f in os.listdir(W): os.replace(f'{W}/{f}', f'{INP}/nn-vecdb-fp32/{f}')
for m in ['torch','transformers','vec2text']: del sys.modules[m]
B = nbformat.read('phaseB_compress_one_scheme.ipynb', 4)
bcells = [fix(c.source) for c in B.cells if c.cell_type == 'code'][1:]
for scheme in ['fp32','int8','int4','pq128x8','binary','svd256']:
    t=time.time(); gb = {}
    exec(bcells[0].replace('SCHEME = "int8"', f'SCHEME = "{scheme}"'), gb)
    gb['N_QUERY_EVAL'] = 2000
    for s in bcells[1:]: exec(s, gb)
    r = gb['results']['runs']
    print(scheme, [(x['model'], x['length'], round(x['mean_cos_to_original'],4), x['cos_in_expected_range'], x['retrieval_top10'], x['search_agrees']) for x in r], f'{time.time()-t:.0f}s', flush=True)
print(sorted(os.listdir(W)), len(os.listdir(W)))
