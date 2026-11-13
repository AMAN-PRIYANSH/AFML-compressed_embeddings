# Proves that resume restores every state exactly: before the first resumed epoch trains, compare live states
# with the checkpoint file.
import sys, os, numpy as np, torch
sys.path.insert(0, os.path.abspath(".")); sys.path.insert(0, os.path.abspath("../ext/vec2vec"))
import types
for name in ["beir", "beir.datasets", "beir.datasets.data_loader_hf"]:
    sys.modules[name] = types.ModuleType(name)
sys.modules["beir.datasets.data_loader_hf"].HFDataLoader = object
import train as T
from v2v_runner import run
ck = torch.load("t2/resume_s1.pt", map_location="cpu", weights_only=False)
real = T.training_loop_
done = {}
def eq(a, b):
    if isinstance(a, torch.Tensor): return torch.equal(a.cpu(), b.cpu())
    if isinstance(a, dict): return a.keys() == b.keys() and all(eq(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)): return len(a) == len(b) and all(eq(x, y) for x, y in zip(a, b))
    return a == b
def probe(*a, **k):
    if not done:
        acc = k["accelerator"]
        done["translator"] = eq(acc.unwrap_model(k["translator"]).state_dict(), ck["translator"])
        done["gen_opt"] = eq([o.state_dict() for o in k["gen_optimizers"]], ck["gen_optimizers"])
        done["gen_sched"] = eq([s.state_dict() for s in k["gen_schedulers"]], ck["gen_schedulers"])
        for n in ck["gans"]:
            g = k[n]
            done[n + "_disc"] = eq(acc.unwrap_model(g.discriminator).state_dict(), ck["gans"][n]["disc"])
            done[n + "_opt"] = eq(g.discriminator_opt.state_dict(), ck["gans"][n]["opt"])
        done["sched_step"] = k["gen_schedulers"][0].state_dict().get("last_epoch")
        print("RESUME CHECK", done, flush=True)
    return real(*a, **k)
T.training_loop_ = probe
d = "t2/data"; L = lambda f: np.load(f"{d}/{f}")
Xv, Yt, Ya = L("bge-large_victim_L32.npy"), L("gtr_victim_L32.npy"), L("gtr_attextra_L32.npy")
idx = np.random.default_rng(0).choice(50000, 8192, replace=False)
run("../ext/vec2vec", Xv[:5000], Ya[:20000], Xv[idx][:1024], Yt[idx][:1024], "t2/runs/check", budget_hours=0.045,
    seed=5, val_size=1024, extra_args={"mixed_precision": "no", "val_bs": 512, "top_k_size": 512, "top_k_batches": 2},
    early_stopping=False, resume_from="t2/resume_s1.pt")
