# Phase V on the workstation: original vec2vec, up to 3 days per run

Neural Nexus, vector-DB compression project. Same experiment as the Kaggle notebook `phaseV_vec2vec_budget.ipynb`:
same official vec2vec code (commit 5efb3c0) and settings, same data files, seed 5, same 8,192 scoring passages.
Only the machine, the time budget and the folder paths differ. The summary file records the GPU.

Files in this folder:
- `run_v2v_local.py`: the run itself (loads data, trains, scores every epoch, saves, plots)
- `v2v_runner.py`: the shared vec2vec wrapper (identical to the one inside the Kaggle notebook)
- `start_run.ps1`, `status.ps1`, `stop_run.ps1`: start in background / check / stop
- `backup_loop.ps1`: copies results to a cloud-synced folder every hour

## Setup (once, about 20-30 minutes)

All commands are typed in **PowerShell** (Start menu -> type PowerShell -> open). No admin rights needed.

1. Put this folder at `C:\Users\Student5\ml-workspace\vec2vec-run`, then:

       cd C:\Users\Student5\ml-workspace\vec2vec-run

2. Reuse the working vec2text environment's versions (same torch cu132 build, nothing re-downloaded from the torch index):

       copy ..\vec2text-env\pyproject.toml .
       copy ..\vec2text-env\uv.lock .
       uv sync
       uv add toml wandb evaluate diffusers nltk seaborn matplotlib scikit-learn scipy

   Expected: uv prints the packages it added; torch stays `2.15.0+cu132`.
   Check: `uv run python -c "import torch; print(torch.__version__, torch.cuda.get_device_name(0))"`
   should print `2.15.0+cu132 NVIDIA RTX 4500 Ada Generation`.

3. Get the official vec2vec code at the fixed version:

       git clone https://github.com/rjha18/vec2vec.git vec2vec
       cd vec2vec
       git checkout 5efb3c02ee908847581c6488cf9c8ed6dace3c29
       cd ..

4. Download the data files from Kaggle (browser -> notebook page -> Output tab -> download icon next to the file)
   into `C:\Users\Student5\ml-workspace\vec2vec-run\data\`:

   | File | From notebook | Size |
   |---|---|---|
   | `bge-large_victim_L32.npy` | phaseA_fp32_embeddings | 205 MB |
   | `gtr_victim_L32.npy` | phaseA_fp32_embeddings | 154 MB |
   | `gtr_attextra_L32.npy` | phaseA2b_verify_attacker_extra | 614 MB |
   | `decoded_victim_pq128x8_bge-large_L32.npy` | phaseB-pq128x8 (when it is done) | 205 MB |

   Disk: about 1.2 GB for data, about 0.6 GB for results. You have ~37 GB free.

5. Smoke test in the foreground (5-10 minutes, small subset, proves everything works on this machine):

       uv run python run_v2v_local.py --scheme fp32 --hours 0.1 --smoke

   Expected: a line `scheme=fp32 gpu=NVIDIA RTX 4500 Ada Generation ...`, a few lines starting with
   `{"kind": "ours", "epoch": ...`, then a line starting with `DONE`. Note the `epoch_minutes` value.

## Real runs (background, keep going if the window is closed)

    powershell -ExecutionPolicy Bypass -File .\start_run.ps1 -Scheme fp32 -Hours 66
    powershell -ExecutionPolicy Bypass -File .\start_run.ps1 -Scheme pq128x8 -Hours 66     (once the PQ file is in data\)

Both runs share the GPU (each needs only a few GB of its 24 GB). 66 hours = 2.75 days, so both finish inside the
one-week access window with time left for one rerun. A run can end earlier: the authors' early-stopping rule stops it
after epoch 80 if it has not improved for 20 epochs.

Check any time:

    powershell -ExecutionPolicy Bypass -File .\status.ps1

Stop one early (files saved so far are kept):

    powershell -ExecutionPolicy Bypass -File .\stop_run.ps1 -Scheme fp32

## Keeping it safe

- The run tells Windows not to sleep while it is running (no admin needed). The screen may still turn off; that is fine.
- It does NOT survive a restart or a sign-out of the Windows user. Closing UltraViewer or the PowerShell window is fine.
- Backup every hour to a cloud-synced folder (start it after the runs have started):

      powershell -ExecutionPolicy Bypass -Command "Start-Process powershell -WindowStyle Hidden -ArgumentList '-ExecutionPolicy','Bypass','-File','backup_loop.ps1','-Target','C:\Users\Student5\OneDrive\vec2vec-backup'"

- When a run is finished: upload its `outputs\<scheme>` folder to Kaggle as a private dataset (kaggle.com -> Create -> New Dataset),
  name `nn-vecdb-phaseV`, and share it with the Neural Nexus group. Phase D needs the trained translators.

## If something fails

Do not edit the code. Open Claude Code in this folder and paste:

> This folder runs the official vec2vec trainer (github.com/rjha18/vec2vec, commit 5efb3c0) on precomputed vectors
> for the Neural Nexus vector-DB compression project, on Windows 11, non-admin, uv environment (torch 2.15.0+cu132,
> transformers 4.57.6, vec2text 0.0.13). Read README.md, run_v2v_local.py and v2v_runner.py, then
> outputs\<scheme>\stderr.txt and stdout.txt. Find why it failed and explain the cause in plain English first.
> Make the smallest possible change. Do not change the vec2vec hyper-parameters, the seed, the data files, the
> 8,192 evaluation passages or the metric, and do not remove any assert. If the fix needs a design change, stop and tell me.
