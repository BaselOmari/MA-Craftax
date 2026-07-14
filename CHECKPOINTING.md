# Checkpointing & Resume

The separate-IPPO trainer (`baselines/seperate_ippo_rnn.py`) saves its progress to disk so a long run can survive being killed, preempted, or hitting a wall-time limit. To resume, launch it again with the same config; it picks up where it stopped (model weights, optimizer state, and the same Weights & Biases run).

## Quick start

Checkpointing is on by default:

```sh
python baselines/seperate_ippo_rnn.py --config_file seperate_ippo_rnn.yaml
```

If the run stops, rerun the same command. With `RESUME: "auto"` (the default) it resumes from the latest checkpoint, or starts fresh if there is none.

## Configuration parameters

Set these in your config YAML under `baselines/config/`. All are optional.

| Parameter | Type | Default | Meaning |
|---|---|---|---|
| `CHECKPOINTING` | bool | `true` | Turn checkpointing on or off. `false` disables both saving and resuming. |
| `CHECKPOINT_DIR` | string | `""` | Where checkpoints are stored. Empty means `<OUTPUT_DIR or .>/<RUN_NAME>/checkpoints`. Use the same path every launch, or the run can't find the checkpoint to resume from. An absolute path is recommended. |
| `CHECKPOINT_INTERVAL_BLOCKS` | int ≥ 1 | `1` | Save a checkpoint every N logging phases (see below). Higher values save less often. |
| `CHECKPOINT_MAX_TO_KEEP` | int ≥ 1 | `3` | Keep this many of the most recent checkpoints; older ones are deleted. |
| `RESUME` | `"auto"` \| `true` \| `false` | `"auto"` | `"auto"`: resume if a checkpoint exists, otherwise start fresh. `true`: require a checkpoint and resume from it (error if none exists). `false`: always start fresh, ignoring any existing checkpoint. |
| `MAX_BLOCKS_THIS_RUN` | int ≥ 0 | `0` | `0` = run to completion. `>0` = checkpoint and exit after this many logging phases, so a scheduler can requeue the job and it resumes automatically. |

### Logging phases

Training runs in repeated **logging phases**; each phase is `LOGGING_UPDATES_INTERVAL` training updates. A checkpoint is written every `CHECKPOINT_INTERVAL_BLOCKS` phases (and once at the very end). A run has about `TOTAL_TIMESTEPS ÷ (NUM_ENVS × NUM_STEPS × LOGGING_UPDATES_INTERVAL)` phases — lower `LOGGING_UPDATES_INTERVAL` if you want more frequent checkpoints.

### Example

```yaml
CHECKPOINTING: true
CHECKPOINT_DIR: "/n/netscratch/your_lab/runs/exp1/checkpoints"
CHECKPOINT_INTERVAL_BLOCKS: 1
CHECKPOINT_MAX_TO_KEEP: 3
RESUME: "auto"
MAX_BLOCKS_THIS_RUN: 0
```

## Resuming

Launch the run; if it stops, launch the same config again. `RESUME: "auto"` resumes from the latest checkpoint (you'll see a `[resume] ...` line at startup). To restart from scratch instead, set `RESUME: false` or point `CHECKPOINT_DIR` at an empty directory.

If you resume with a config that changed the model size, number of agents, or environment dimensions, the run stops with a clear error instead of loading a mismatched checkpoint — start a fresh run in that case.

## Wall-time-limited jobs (SLURM)

To keep a multi-day run inside shorter time slots (both rely on `RESUME: "auto"`):

- Submit with `#SBATCH --signal=B:USR1@180` and `#SBATCH --requeue`: the run checkpoints and exits when signaled before the wall-time, and SLURM requeues it to continue.
- Or set `MAX_BLOCKS_THIS_RUN` so each launch stops after a fixed number of phases, and submit the job repeatedly; each launch resumes from the last checkpoint.
