# Nativeval RFD3 partial diffusion

This directory keeps input preparation separate from sampling. Preparation copies the
180 validated role-specific semantic CIFs into `original_samples/`, validates them with
RFD3's `inference_load_`, and writes one native RFD3 JSON for every requested condition.
Sampling consumes exactly one prepared JSON, fixes the context ligand, and writes the
entire predicted complex after a strict binder-C-alpha Kabsch alignment to its original.

Prepare the current nativeval partial-t matrix:

```bash
bash prepare_nativeval_inputs.sh \
  --output-root /scratch/users/zhkim216/datasets/evaluation_datasets/nativeval/ensembles
```

Run one role across those six conditions on an interactive A100 node:

```bash
bash run_partial_diffusion_interactive.sh \
  --output-root /scratch/users/zhkim216/datasets/evaluation_datasets/nativeval/ensembles \
  --role-id 8r5n_binder_A_1_context_C_1 \
  --num-samples 1
```

For the current fixed-sequence run, `input_index.txt` has 540 JSON paths. Submit the
54 zero-based shard tasks and pass the number of designs per role-condition explicitly:

```bash
CONTAINER_SHA256=$(sha256sum "$FOUNDRY_SIF" | awk '{print $1}')
CHECKPOINT_SHA256=$(sha256sum "$CKPT_PATH" | awk '{print $1}')
export CONTAINER_SHA256 CHECKPOINT_SHA256 NUM_SAMPLES=5
sbatch --array=0-53%20 run_partial_diffusion.sbatch
```

Each output directory contains the unchanged original `{role_id}.cif`, aligned sample
CIFs named with `seqfix`/`sequnfix`, `ligfix`, and `partialt*`, plus `manifest.json`.
Raw RFD3 outputs are deleted only after all aligned CIFs and the manifest validate; a
failed run prints and preserves its raw directory under `.partial_diffusion_raw/`.

## Denovoval 32-sample ensembles

The denovoval workflow freezes the 3400 CIFs under
`/scratch/users/zhkim216/datasets/evaluation_datasets/denovoval/cifs`, prepares
fixed-sequence/fixed-ligand inputs at partial-t 2 and 5, and maps one array task
to each of the 154 CCD codes. Each task handles both lengths and resumes by skipping
input-conditions that already have a complete 32-sample manifest.

The two conditions produce 6,800 input-conditions and up to 217,600 sample CIFs.
The one-CIF smoke covers both conditions and therefore generates 64 sample CIFs.

The submitter creates a preparation job, a one-CIF smoke job, the dependency-gated
154-task full array, and an `afterany` audit job:

```bash
bash submit_denovoval_partial_diffusion.sh --dry-run
bash submit_denovoval_partial_diffusion.sh
```

Generation uses the `bioe,owners` partitions and A40/A100/H100/H200 GPUs, with an
eight-hour limit and scheduler requeue enabled. Individual application failures are
recorded and skipped. The final audit writes `generation_summary.json`,
`retry_index.txt`, and `FAILURES.md`; it checks manifest counts and declared file
existence without reparsing or rehashing generated CIFs.

To preserve active CCD tasks while replacing only pending work with one-hour chunks,
preview and then execute the live-array reroute:

```bash
bash reroute_pending_denovoval_1h.sh
bash reroute_pending_denovoval_1h.sh --execute
```

The reroute holds the original array, keeps every non-pending task, groups four CIFs
(eight t2/t5 input-conditions) per new task without crossing CCD boundaries, and
submits the replacement array with a one-hour limit. Scheduler interruption uses the
existing requeue/resume contract; user-initiated `scancel` is not retried.
