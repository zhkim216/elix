# Operational Wrapper Role

Use this role for local launchers, Sherlock sbatches, smoke wrappers, and command adapters.

## Implementation Shape

- Start from the final runtime command and existing entrypoint.
- Keep wrappers thin: set environment, choose paths/configs, echo the command, then call the entrypoint, preferably with `python -m` for repo modules.
- Before adding a wrapper function, copied entrypoint, dataclass/config object,
  or Python orchestrator for a Sherlock/local workflow, check whether the
  existing entrypoint can be called directly with explicit CLI, Hydra, or sbatch
  values. Prefer editing that command block unless a concrete unmet requirement
  needs a new layer.
- For existing eval, sampling, or training entrypoints, prefer direct entrypoint
  execution plus config, Hydra override, or sbatch values. Create a new Python
  orchestrator only when the existing entrypoint cannot express the workflow,
  multiple independent workflows need the same non-entrypoint logic, or the user
  explicitly asks for it. Scheduler arrays, checkpoint sweeps, model-name loops,
  and smoke/full-run variants are parameterization, not call sites.
- Valid wrapper responsibilities are environment setup, path/config selection,
  scheduler matrices, command echoing, and bounded smoke defaults. Avoid
  schema expansion, generic submit APIs, alternate execution paths, or embedding
  provenance fields in runtime inputs unless the consumer contract requires
  them. Separate prepare/annotation utilities may write sidecar manifests or
  validation reports when they do not change the runtime input schema.
- For self-contained sbatch launchers, keep experiment-defining values in one
  visible block. Derive run IDs, output filenames, list paths, array dimensions,
  and labels from that block; do not duplicate the same selection in YAML,
  hardcoded names, and command args unless one artifact is clearly generated
  from the other.
- Before adding or changing `EXP_NAME`, `RUN_NAME`, `HYDRA_ROOT`, output
  prefixes, or scheduler labels, lock the output naming contract. Classify each
  sweep value as a scheduler selector, a consumer selector, or a top-level run
  identity. If a value is already passed to the consumer as a selector, such as
  `model_cfg.ckpt_cfg.start_step`, `pdb_cfg.array_id`, `sample_lengths`,
  `guidance_scale`, or `num_samples`, first inspect whether the consumer or a
  sibling launcher already creates the corresponding subfolder, record key, or
  metrics partition. Do not duplicate that value in the top-level experiment
  name unless the closest same-output-hierarchy convention does so.
- Treat copied derived labels as suspicious. When adapting a single-step,
  single-model, or one-off launcher into a sweep, compare against the nearest
  launcher with the same output hierarchy before preserving labels like
  `step${STEP}`, `chunk${CHUNK_IDX}`, or `sample${SAMPLE_IDX}` in `EXP_NAME`.
  Prefer names that identify the run family and fixed knobs; let scheduler and
  checkpoint selectors partition outputs through the existing consumer contract.
- Separate local and Sherlock assumptions explicitly.
- Make debug defaults bounded; do not make a full sweep the default unless the user explicitly requests it.

## Validation

- Compare against a nearby known-working launcher when available.
- Run `bash -n` for shell/sbatch files.
- Use dry-run, echo-only, or tiny smoke commands before expensive jobs.
- Make experiment-defining switches explicit instead of relying on hidden defaults.
- For launchers that derive names or selections, validate the resolved
  identifiers and selected input count when feasible, and report examples such
  as run ID, experiment name, generated list path, and selected sample count.
- If a prepare utility generates runtime inputs, run it on a tiny representative
  fixture and validate the generated artifact through the existing entrypoint's
  reader, Hydra composition, dry-run path, or closest pure helper.
