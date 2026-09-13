# Working instructions

This repository is edited on the user's Mac and compute-heavy work is run on the MPI-IS HTCondor cluster. Do not run training or other GPU-heavy commands on the local Mac or on a cluster login node.

## Locations and access

- Local clone: `/Users/fdraye/Library/CloudStorage/OneDrive-Personal/PC/cluster/toy_model_linear`
- GitHub: `git@github.com:florentdraye/toy_model_linear.git`
- Cluster login: `ssh -tt fdraye@login.cluster.is.localnet`
- Cluster clone: `/lustre/home/fdraye/projects/toy_model_linear`
- The public hostname must be used from the Mac; `login1`, `login4`, etc. are internal node names and do not resolve locally.
- SSH may request the passphrase for `~/.ssh/id_rsa`. Never ask the user to paste it into chat. Ask them to run `ssh-add ~/.ssh/id_rsa` in their own Mac terminal, enter it there, and then retry.

## Normal workflow

1. Inspect and edit the local clone.
2. Run only lightweight local checks when useful.
3. Commit and push changes from the local clone when the user requests synchronization or a cluster run depends on them.
4. Connect to the cluster, enter the cluster clone, and pull the intended commit before running:

   ```bash
   ssh -tt fdraye@login.cluster.is.localnet
   cd /lustre/home/fdraye/projects/toy_model_linear
   git pull --ff-only
   ```

5. Never overwrite uncommitted cluster changes. Check `git status --short --branch` before pulling.

## Interactive GPU session

The project includes `interactive_gpu.sub`, requesting one NVIDIA H100 (CUDA capability 9.0), 10 CPUs, and 100 GB RAM. The cluster requires a compute-unit bid, so plain `condor_submit -i interactive_gpu.sub` is rejected. From the cluster project directory use:

```bash
condor_submit_bid 2000 -i interactive_gpu.sub
```

Wait for the interactive shell to attach, then verify that the prompt is on a compute node and run:

```bash
pwd
hostname
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
python3 -c 'import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))'
```

A verified session on 2026-07-20 provided an NVIDIA H100 with about 95 GB VRAM and PyTorch 2.7.0+cu128. Treat those versions as observations, not permanent guarantees.

The cluster login banner says not to launch compute-intensive work on login nodes. Submit batch work with the repository's `submit_*.sub` files and `condor_submit_bid`; inspect the chosen submit file and experiment script before launch. Existing 2k submit files document a bid of 2000.

## Operational notes

- Include the actual plot inline with experimental results so the user can verify them visually; a report link alone is insufficient.
- Steering location is not restricted to the last token: the user wants steering evaluated across locations and the best token position shown.
- Interactive allocations and SSH sessions do not survive assistant sessions reliably; reconnect and request a fresh allocation when necessary.
- Report the Condor job/cluster ID, compute hostname, GPU, command launched, and output/run directory to the user.
- Avoid committing generated artifacts. `runs/`, `logs/`, checkpoints, plots, and caches are intentionally ignored; confirm with `.gitignore` before adding files.
- Read `README.md`, `train.py`, and the selected `run_*.sh`/`submit_*.sub` recipe before changing or launching an experiment.
