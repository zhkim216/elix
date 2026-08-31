# SwitchCraft Two-Lane Generation Plan

## Goal

Generate the frozen 26,023-design SwitchCraft benchmark while using Sherlock GPU capacity efficiently and preserving the scientific manifest at:

`/scratch/users/zhkim216/experiment_result_analysis/benchmarks/state_switching/denovoval/manifests/manifest.json`

All generated structures remain under:

`/scratch/users/zhkim216/experiment_result_analysis/benchmarks/state_switching/denovoval`

The requested dataset is:

| Family | States | Designs |
|---|---:|---:|
| Positive allostery | 2 | 12,000 |
| Negative allostery | 2 | 12,000 |
| Motif switching | 2 | 1,000 |
| Ligand modification | 2 | 558 |
| Ligand discrimination | 3 | 465 |
| **Total** |  | **26,023** |

Every state uses the default SwitchCraft configuration and five diffusion samples. The controller changes only scheduling and task grouping; it does not change the scientific configuration.

## GPU lanes

Each immutable batch contains at most 1,000 Slurm array tasks. The `gpu`
partition is deliberately excluded from both lanes so its 100-job submit limit
does not constrain the batch.

| Lane | GPU constraints | Partitions | Time | Designs/task | Array cap | Resources |
|---|---|---|---:|---:|---:|---|
| Slow | A40, A100 PCIe/SXM4, H100 SXM5, H200 SXM5 | bioe,possu,owners | 1 h | 2 for 2-state only | 4 | 2 CPU, 16 GiB, 1 GPU |
| Fast | A40, A100 PCIe/SXM4, H100 SXM5, H200 SXM5 | bioe,possu,owners | 2 h | 10 for 2-state; 5 for 3-state | 10 | 2 CPU, 32 GiB, 1 GPU |

The first batch has up to 450 slow tasks and 550 fast tasks. All 465 three-state designs fit in 93 fast tasks; the remaining fast tasks and all slow tasks are filled with two-state designs. Two-state assignments are selected in case order using repeated round-robin passes, so early manifest families do not monopolize a batch.

The combined concurrency is 14 array elements: four slow and ten fast. At two
CPUs per element, the two arrays use at most 28 CPUs concurrently. The `normal`
partition itself has no GPUs and is used only by the CPU finalizer.

## Immutable manifests and completion contract

`two_lane_generation.py prepare` first scans existing outputs. A design counts as complete only when all expected state pickle files, five PDB/CIF sample pairs per state, and required motif specification pickles are present and structurally readable. Already complete designs are frozen in `initial_complete.tsv` and never assigned.

Each batch receives immutable explicit design-ID manifests:

`manifests/two_lane_1h2h_v1/batches/batch_NNN/{slow.tsv,fast.tsv,batch.json}`

Workers consume those explicit IDs. They re-check artifacts before each design and skip a design only if the strict artifact contract passes. GPU compilation caches are separated by the actual GPU SKU.

## Retry and automatic continuation

Each slow/fast submission has a deterministic job name and an atomic submission receipt. Before submitting, the controller reconciles the name against `squeue` and `sacct`, preventing duplicate submissions after a controller crash.

An `afterany` CPU finalizer validates artifacts and records Slurm states. Artifact validity decides completion; Slurm state explains the failure. Missing or partial design directories are moved to the recoverable run quarantine before retry. Timeout, preemption, node failure, and ordinary nonzero exits are retried automatically for only the incomplete task IDs, with at most five total attempts. `OUT_OF_MEMORY`, `CANCELLED`, missing accounting state, and exhausted attempts block automatic progress for inspection.

When a batch validates cleanly, the finalizer freezes and submits the next batch.
After the last batch, the controller runs a complete 26,023-design validation and
writes `final_validation.json`, including missing/invalid counts and examples.

## Transition from the legacy arrays

Legacy jobs `34762517` and `34762519` are transitioned without killing useful work:

1. Cancel only their `PENDING` elements.
2. Allow their currently `RUNNING` elements to finish.
3. Run the initial artifact scan after those elements are terminal.
4. Freeze and submit batch 000 from the still-incomplete designs.

The old mutable `work_items.tsv` is not overwritten while legacy elements can still read it.
The transition is implemented as a `normal`-partition CPU job with an `afterany`
dependency on both legacy array roots. It therefore starts the scan only after
all surviving legacy elements are terminal, and its dry run is immediately
followed by the exactly-once controller submission.

## Pause for exp37 nativeval

The active batch-000 submission was intentionally cancelled to release its GPUs
for the exp37 nativeval campaign managed by watcher job `34851148`. Completed
SwitchCraft artifacts remain valid and are not deleted.

`start_two_lane_after_exp37_nativeval.sbatch` is submitted with an `afterany`
dependency on that watcher. It fails closed unless the watcher itself is
`COMPLETED` and the frozen exp37 status artifacts report all 36 targets and all
144 chunks as exactly `complete`, with no blocked, active, waiting, or
metric-error state. Only after that gate passes does it run batch 000 attempt 1
finalization with the explicit `--retry-cancelled` option. This preserves valid
outputs and submits only incomplete task IDs as attempt 2. Ordinary automatic
finalizers continue to block on `CANCELLED` unless an operator explicitly uses
that option. If Slurm stores never-started cancelled array elements only as a
compressed range, the transition also uses `--retry-missing-accounting`; this
still requires artifact validation to mark the exact task incomplete, while
ordinary automatic finalizers remain fail-closed on missing accounting rows.

The first dependent transition (`34852581`) passed the exp37 gate but stopped
because Slurm represented 890 never-started cancelled elements only as
compressed array ranges, without individual `sacct` rows. After a
production-root dry run validated the exact incomplete task sets, batch 000 was
resumed as attempt 2 on 2026-07-26: slow `35900506`, fast `35900513`, and
finalizer `35900515`. The exact receipt is recorded in
`manifests/two_lane_1h2h_v1/transition_after_exp37.json`.

## Operator commands

Prepare and freeze the first batch:

```bash
python -B scripts_jinho/two_lane_generation.py prepare
```

Inspect the exact submissions without changing scheduler state:

```bash
python -B scripts_jinho/two_lane_generation.py submit --batch-index 0 --attempt 1 --dry-run
```

Submit batch 000:

```bash
python -B scripts_jinho/two_lane_generation.py submit --batch-index 0 --attempt 1
```

Inspect controller receipts and active scheduler rows:

```bash
python -B scripts_jinho/two_lane_generation.py status
```

Monitoring is performed at 15-minute intervals. Each report checks the exp37
completion gate until it passes, then lane counts and states, failure reasons,
finalizer state, artifact progress, retry/quarantine diagnostics, and whether
the next batch was submitted exactly once.
