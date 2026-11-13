import numpy as np, json, random, pandas as pd
r=np.random.default_rng(0); random.seed(0)
W="Paris France London John Smith England Berlin Germany the a of in war king music album song team cup".split()
NV, NA, NE, N = 5000, 1500, 1024, 200
texts={"victim_p_L32":[" ".join(random.choice(W) for _ in range(12))+f" x{i}" for i in range(NV)]}
json.dump(texts,open("tE/in/A/texts.json","w"))
np.save("tE/in/A/gtr_victim_L32.npy",(r.normal(size=(NV,768))+3).astype(np.float32))
np.save("tE/in/A/bge-large_victim_L32.npy",r.normal(size=(NV,1024)).astype(np.float32))
np.save("tE/in/A/bge-large_att_L32.npy",r.normal(size=(NA,1024)).astype(np.float32))
np.save("tE/in/A/gtr_att_L32.npy",(r.normal(size=(NA,768))+3).astype(np.float32))
keep=np.ones(NA,bool); keep[:50]=False; np.save("tE/in/C2/attacker_keep_mask.npy",keep)
np.save("tE/in/C2/mean_A_fp32.npy",np.zeros(1024,np.float32)); np.save("tE/in/C2/mean_B.npy",np.full(768,3,np.float32))
eval_idx=np.random.default_rng(0).choice(NV,NE,replace=False); pick=eval_idx[:N]
truth=[texts["victim_p_L32"][i] for i in pick]
for s in ["fp32","int8","int4","pq128x8","binary","svd256"]:
    runs=[{"scheme":s,"seed":k,"refine2":{"top1":float(r.uniform()),"mean_cos":0.4},"self_consistency":float(r.uniform())} for k in range(5)]
    best=max(runs,key=lambda x:x["self_consistency"])
    json.dump({"runs":runs,"phaseD_pick":{s:{"seed":best["seed"]}},"norm_B":27.0},open(f"tE/in/C2/results_C_{s}.json","w"))
    drop = 3 if s in ("pq128x8","binary") else 2
    rec=[" ".join(w for k,w in enumerate(t.split()) if k%drop) for t in truth]
    pd.DataFrame({"true":truth,"recovered":rec,"token_f1":0,"phaseC_hit":True}).to_csv(f"tE/in/D/inversions_{s}.csv",index=False)
    np.save(f"tE/in/C2/translated_eval_{s}.npy",(r.normal(size=(NE,768))+3).astype(np.float32))
np.save("tE/in/E1/translated_eval_sup_mlp.npy",(r.normal(size=(NE,768))+3).astype(np.float32))
row=lambda f:{"token_f1":f,"bleu":1.0,"entity_recovery":0.1}
json.dump({"schemes":{"fp32":row(0.24)},"controls":{"positive_true_vectors":row(0.95),"floor_random_other_passage":row(0.1)}},open("tE/in/D/results_D.json","w"))
print("fake ok")
