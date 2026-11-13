"""Run the OFFICIAL vec2vec trainer (github.com/rjha18/vec2vec, commit 5efb3c0) on our saved vectors.

What is changed, and nothing else:
  1. Data: instead of embedding text on the fly, the trainer gets our saved vectors through its own
     "precomputed embeddings" path (the one the authors use for fMRI data). Victim side = BGE vectors
     (fp32 or decoded compressed), attacker side = GTR vectors. No pairs are given to training.
  2. Encoder sizes: that path assumes both spaces have the same size; ours are 1024 (BGE) and 768 (GTR).
  3. A wall-clock budget: training stops cleanly when the budget is used up.
  4. After every epoch we score the translation with OUR metric (same one as mini-vec2vec):
     top-1 / average rank / mean cosine of victim vectors translated into GTR space vs their true GTR vectors.
     The true GTR vectors are used only for this scoring, never for training.
  5. wandb logging is replaced by a local JSON-lines log.
All losses, architecture, optimiser and hyper-parameters come from the authors' configs/unsupervised.toml.
"""
import os, sys, json, time, types, inspect, shutil

def run(repo_dir, X_victim, Y_attacker, X_eval, Y_eval, out_dir, budget_hours,
        seed=5, val_size=4096, extra_args=None, log=print):
    import numpy as np, torch
    t_start = time.time()
    os.makedirs(out_dir, exist_ok=True)
    sys.path.insert(0, repo_dir)
    os.chdir(repo_dir)

    # beir is only used by the text-dataset path we do not use; stub it so it need not be installed
    if "beir" not in sys.modules:
        for name in ["beir", "beir.datasets", "beir.datasets.data_loader_hf"]:
            sys.modules[name] = types.ModuleType(name)
        sys.modules["beir.datasets.data_loader_hf"].HFDataLoader = object

    # Windows: vec2text imports the Unix-only "resource" module; give it a harmless stand-in
    if os.name == "nt" and "resource" not in sys.modules:
        try:
            import resource  # noqa: F401  (already provided, e.g. by a shim)
        except ImportError:
            stub = types.ModuleType("resource")
            stub.RLIMIT_NOFILE = 7
            stub.getrlimit = lambda *_: (8192, 8192)
            stub.setrlimit = lambda *_: None
            sys.modules["resource"] = stub

    import wandb
    import train as T
    from fmri.data import FmriSingleViewDataset

    # Windows: data-loader worker processes use "spawn" and would have to copy the whole dataset;
    # their helper also uses a Linux-only call. Load data in the main process instead (no speed
    # loss here: the data are already tensors in memory).
    if os.name == "nt":
        T.get_num_proc = lambda: 0

    class PairedDifferentSizes(torch.utils.data.Dataset):
        """Same as the authors' FmriPairedDataset, but the two spaces may have different sizes."""
        def __init__(self, name_a, Za, name_b, Zb):
            assert len(Za) == len(Zb)
            self.a, self.Za, self.b, self.Zb = name_a, Za, name_b, Zb
        def __len__(self):
            return len(self.Za)
        def __getitem__(self, i):
            return {self.a: self.Za[i], self.b: self.Zb[i]}

    if not torch.cuda.is_available():          # their helper divides by the GPU count (CPU-only test runs)
        T.get_num_proc = lambda: 2

    # -- 5. local logging instead of wandb
    log_path = os.path.join(out_dir, "v2v_log.jsonl")
    def _wandb_log(d, *a, **k):
        rec = {"t_hours": (time.time() - t_start) / 3600}
        for key, v in d.items():
            if isinstance(v, (int, float)):
                rec[key] = float(v)
        with open(log_path, "a") as f:
            f.write(json.dumps({"kind": "authors_val", **rec}) + "\n")
    wandb.log = _wandb_log
    wandb.Image = lambda *a, **k: None

    # -- 1. our data through the precomputed path
    Xv = torch.tensor(np.asarray(X_victim, dtype=np.float32))
    Ya = torch.tensor(np.asarray(Y_attacker, dtype=np.float32))
    Xe = torch.tensor(np.asarray(X_eval, dtype=np.float32))
    Ye = torch.tensor(np.asarray(Y_eval, dtype=np.float32))
    def _loader(*a, **k):
        supset = FmriSingleViewDataset("gtr", Ya)          # attacker's own space ("supervised" = known model)
        unsupset = FmriSingleViewDataset("bge", Xv)        # stolen victim vectors ("unsupervised" = unknown model)
        n = min(val_size, len(Xe))
        val = PairedDifferentSizes("gtr", Ye[:n], "bge", Xe[:n])
        return supset, unsupset, val, {"gtr": Ya.shape[1], "bge": Xv.shape[1]}
    T.load_fmri_train_and_eval_datasets = _loader

    # -- 4. our metric after every epoch
    Xe_n = torch.nn.functional.normalize(Xe, dim=1)
    Ye_n = torch.nn.functional.normalize(Ye, dim=1)
    best = {"top1": -1.0}
    state = {"epoch": 0}
    def _score(translator):
        dev = next(translator.parameters()).device
        was_training = translator.training
        translator.eval()
        outs = []
        with torch.no_grad():
            for i in range(0, len(Xe_n), 1024):
                ins = {"bge": Xe_n[i:i+1024].to(dev), "gtr": Ye_n[i:i+1024].to(dev)}
                _, tr = translator(ins, include_reps=False)
                outs.append(tr["gtr"]["bge"].float().cpu())
        if was_training:
            translator.train()
        Z = torch.nn.functional.normalize(torch.cat(outs), dim=1)
        S = Z @ Ye_n.T
        d = S.diag()
        ranks = (S > d[:, None]).sum(1) + 1
        return {"top1": float((ranks == 1).float().mean()), "avg_rank": float(ranks.float().mean()),
                "mean_cos": float(d.mean())}

    orig_loop = T.training_loop_
    def _loop(*a, **k):
        t0 = time.time()
        r = orig_loop(*a, **k)
        state["epoch"] += 1
        sc = _score(k["translator"])
        rec = {"kind": "ours", "epoch": state["epoch"], "t_hours": (time.time() - t_start) / 3600,
               "epoch_minutes": (time.time() - t0) / 60, **sc}
        with open(log_path, "a") as f:
            f.write(json.dumps(rec) + "\n")
        log(json.dumps(rec))
        # crash safety: keep the latest and the best translator outside the authors' save_dir
        model = k["accelerator"].unwrap_model(k["translator"])
        torch.save(model.state_dict(), os.path.join(out_dir, "translator_latest.pt"))
        if sc["top1"] > best["top1"]:
            best.update(sc, epoch=state["epoch"])
            torch.save(model.state_dict(), os.path.join(out_dir, "translator_best.pt"))
            with open(os.path.join(out_dir, "best.json"), "w") as f:
                json.dump(best, f)
        return r
    T.training_loop_ = _loop

    # -- 2 + 3. patch two lines of main(): per-space encoder sizes, and the time budget
    src = inspect.getsource(T.main)
    a = "encoder_dims = {cfg.sup_emb: int(embed_dim), cfg.unsup_emb: int(embed_dim)}"
    b = "encoder_dims = {cfg.sup_emb: int(embed_dim[cfg.sup_emb]), cfg.unsup_emb: int(embed_dim[cfg.unsup_emb])}"
    c = "    for epoch in range(max_num_epochs):\n"
    d = c + "        if _TIME_UP():\n            print('Time budget reached, stopping.')\n            break\n"
    e = "drop_last=cfg.train_drop_last if hasattr(cfg, 'train_drop_last') else not fmri_mode"
    f_ = "drop_last=bool(cfg.train_drop_last) if hasattr(cfg, 'train_drop_last') else not fmri_mode"
    g = "wandb.log(val_res)"   # wandb.init() re-binds wandb.log, so route the call explicitly
    assert src.count(a) == 1 and src.count(c) == 1 and src.count(e) == 2 and src.count(g) == 1, \
        "vec2vec code changed: patch points not found"
    src = src.replace(a, b).replace(c, d).replace(e, f_).replace(g, "_WANDB_LOG(val_res)")
    T.__dict__["_WANDB_LOG"] = _wandb_log
    # the budget counts training time from the start of the first epoch (setup time excluded),
    # and at least one epoch always runs
    clock = {"t0": None}
    def _time_up():
        if clock["t0"] is None:
            clock["t0"] = time.time()
            return False
        return (time.time() - clock["t0"]) / 3600 > budget_hours
    T.__dict__["_TIME_UP"] = _time_up
    exec(compile(src, T.__file__, "exec"), T.__dict__)

    args = {"unsup_emb": "bge", "sup_emb": "gtr", "dataset": "fmri_precomputed", "epochs": 100000,
            "use_wandb": 0, "seed": seed, "val_size": val_size, "save_dir": out_dir.rstrip("/") + "/authors_{}/",
            "force_wandb_name": 1, "wandb_name": "run",
            # our two sides have different sizes (50k vs 200k): drop the last partial batch so every
            # step pairs two full batches, as the text path of the authors' code does by default
            "train_drop_last": 1}
    args.update(extra_args or {})
    argv = ["train.py", "unsupervised"]
    for k_, v_ in args.items():
        argv += [f"--{k_}", str(v_)]
    sys.argv = argv
    T.argv = argv
    T.main()
    return best, log_path
