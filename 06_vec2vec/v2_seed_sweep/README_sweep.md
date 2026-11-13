# vec2vec sweep: workstation part (Neural Nexus, vector-DB compression project)

Same code as the Kaggle sweep notebook `phaseV2_vec2vec_sweep.ipynb` (official vec2vec, commit 5efb3c0, the
authors' settings, early stopping OFF, a full checkpoint `resume.pt` every 30 minutes).

New files (copy into `C:\Users\Student5\ml-workspace\vec2vec-run`, next to the old ones; nothing old is deleted):
- `v2v_job.py`: one run (kind, seed, budget). Replaces `run_v2v_local.py` for new runs.
- `v2v_runner.py`: the shared wrapper (new version: checkpoints + resume; still works with the old script).
- `start_job.ps1`, `status.ps1`, `stop_job.ps1`, `backup_loop.ps1`.

## The two runs to start now (about 66 hours each, both on the one GPU)

    powershell -ExecutionPolicy Bypass -File .\start_job.ps1 -Name ours_fp32_s7 -Kind ours -Seed 7 -Hours 66
    powershell -ExecutionPolicy Bypass -File .\start_job.ps1 -Name ours_fp32_s8 -Kind ours -Seed 8 -Hours 66

## Later: continue the sweep's winner from its Kaggle checkpoint

Download `runs/<name>/resume.pt` from the friend's notebook (Output tab) into Downloads, then for example:

    powershell -ExecutionPolicy Bypass -File .\start_job.ps1 -Name ours_fp32_s2 -Kind ours -Seed 2 -Hours 60 -Resume C:\Users\Student5\Downloads\resume.pt

`-Hours` is the new TOTAL (Kaggle's 10.5 h count towards it). Kind and seed must match the checkpoint; the
script refuses a mismatch. A run that is not learning can be stopped any time:

    powershell -ExecutionPolicy Bypass -File .\stop_job.ps1 -Name ours_fp32_s8

## Check / backup

    powershell -ExecutionPolicy Bypass -File .\status.ps1

Backup: the backup loop copies `outputs\` to OneDrive every hour (each resume.pt is about 0.4 GB).
Runs survive closing UltraViewer and Windows "Switch user". They do NOT survive sign-out or restart; then
continue each run with `-Resume outputs\<Name>\resume.pt` (at most 30 minutes lost).
