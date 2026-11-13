import numpy as np, json, random
r = np.random.default_rng(0); random.seed(0)
words = "the city of paris france river john smith 1990 war king england london music album song team world cup".split()
texts = {"victim_p_L32": [" ".join(random.choice(words) for _ in range(15)).capitalize() + f" Item{i}" for i in range(50000)]}
json.dump(texts, open("tD/input/A/texts.json", "w"))
Y = r.normal(size=(50000, 768)).astype(np.float32) + 3
np.save("tD/input/A/gtr_victim_L32.npy", Y)
mB = Y.mean(0); np.save("tD/input/C2/mean_B.npy", mB)
json.dump({"norm_B": 27.0}, open("tD/input/C2/results_C_fp32.json", "w"))
idx = np.random.default_rng(0).choice(50000, 8192, replace=False)
for s, noise in [("fp32", 0.5), ("pq128x8", 3.0)]:
    T = Y[idx] + noise * r.normal(size=(8192, 768)); np.save(f"tD/input/C2/translated_eval_{s}.npy", T.astype(np.float32))
print("fake ok")
