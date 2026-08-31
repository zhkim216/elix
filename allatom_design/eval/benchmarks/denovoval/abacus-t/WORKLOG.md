# Denovoval ABACUS-T execution worklog

This is the append-only operational record for the denovoval ABACUS-T
staging, sampling, CCD backmapping, AF3 self-consistency, and metric gates.
The ABACUS-T released model code is treated as immutable: adapter-side fixes
must preserve its original inference behavior.

## Active contract

- Canonical input: 3,400 denovoval CIF files.
- Sampling: ABACUS-T single-state self-conditioned 650M defaults from
  `model/abacusT_sstate/run.sh`, two designs per input, design indices `0,1`,
  final iteration `19`.
- Ligands: explicit-chemistry SDF is the preprocessing source of truth; staged
  PDB supplies coordinates and protein context. Five-character CCD codes and
  `NA`/`K` use PDB-safe aliases and are restored during CIF backmapping.
- Gates are sequential and fail closed: 2-input smoke sampling, smoke
  backmapping/AF3/metrics, one length-150 input per CCD, full sampling, full
  AF3/metrics.
- Any nonzero missing, stale, failed, unmapped, graph-mismatch, sequence-
  mismatch, or exact-count diagnostic stops the pipeline before the next gate.

## Errors and resolutions

### 2026-07-16: SDF chemistry could be bypassed by PDB-first discovery

- Symptom: the released preprocessor could discover the staged PDB before the
  matching SDF, losing explicit bond order and aromaticity for ligand graphs.
- Cause: candidate-file ordering in the preprocessor path, not CCD naming or
  the ABACUS-T ligand feature schema.
- Resolution: the denovoval adapter copies the validated SDF into each clean
  task shard using the expected `lig_<chain>_<alias>_<resid>_1.sdf` basename
  and gives SDF priority. The released sampler/preprocessor implementation is
  not modified.
- Validation: the task runner requires an exact ligand key, atom/bond counts,
  normalized RDKit graph hash, coordinates, atomic-number/formal-charge
  features, edge shapes, and UniMol atom count before sampling.
- Status: resolved in adapter; smoke runtime validation pending.

### 2026-07-16: monatomic CCD SDF construction produced non-finite alignment

- Symptom: the generic AtomWorks CCD-to-SDF path failed for monatomic ions
  whose CCD mirror component has no finite conformer coordinates.
- Affected CCDs: `CA`, `CO`, `CU`, `FE`, `FE2`, `K`, `MG`, `MN`, `NA`, `NI`,
  `ZN`.
- Cause: a one-atom conformer with missing CCD coordinates reached an SVD-based
  coordinate alignment.
- Resolution: construct a one-atom RDKit molecule directly from the pinned CCD
  element and formal charge, use the staged PDB coordinate, and write zero
  bonds. Re-read the SDF and apply the same graph/coordinate validation.
- Evidence: smoke `NA_len150_0` is complete with one atom, zero bonds, finite
  coordinate error, and `sdf_generation_method=direct_monatomic_ccd`.
- Status: resolved and staging-validated.

### 2026-07-16: producer/AF3 design-index convention mismatch

- Symptom: ABACUS-T emits design indices `0,1`, while the shared AF3 path had a
  legacy default convention of `1..N` for other sequence-design producers.
- Cause: producer convention was hard-coded in the shared consumer.
- Resolution: the shared AF3 runner accepts explicit
  `sequence_design.design_indices`; ABACUS-T declares `[0,1]`, while callers
  that omit the key retain the legacy `1..N` default.
- Status: code implemented; focused regression tests pending in this resumed
  run. These values are design indices, not CUDA device IDs.

### 2026-07-16: declared augmentation differs from effective self-conditioning noise

- Observation: the source `run.sh` declares `augment_eps=0.2`, but the released
  self-conditioning inference path effectively samples with noise scale `0.0`.
- Decision: preserve the released behavior exactly. Do not patch this internal
  ABACUS-T behavior; record both declared and effective values in the task
  contract.
- Status: accepted upstream behavior, not an adapter error.

### 2026-07-16: overly broad temporary-directory cleanup removed `/scratch/.../debug`

- Symptom: cleanup intended for a temporary ABACUS-T consumer smoke directory
  operated on an ancestor path and removed unrelated debug artifacts.
- Cause: destructive cleanup was derived from parent traversal instead of an
  exact allow-listed task path.
- Resolution: sampling cleanup is restricted to the exact canonical
  `sampling/<mode>/tasks/task_NNNN` path, rejects symlinks, and refuses any path
  outside that contract before calling `rmtree`. Existing unrelated debug data
  cannot be fully reconstructed and is outside the resumed pipeline.
- Status: guard present; syntax/unit/runtime checks pending before submission.

## Gate log

### 2026-07-16: resumed-state audit

- Scheduler: no running or historical `abacust_denovoval` job for this run.
- Staging: `staging_manifest_smoke.csv` has exactly two complete inputs,
  `A1L3W_len150_0` and `NA_len150_0`; referenced PDB/SDF hashes match.
- Diagnostics: `failed=0`, `missing=0`, `stale=0`.
- Downstream state: no sampling, backmapped, AF3, or metric artifacts exist.
- Next permitted gate: submit array task `0` for 2-input smoke sampling and
  require exactly four validated design-manifest rows before continuing.

### 2026-07-16: pre-submission validation stopped on missing repo Python

- Passed: targeted Python compilation for the four ABACUS-T repo modules,
  YAML syntax, scratch Python compilation, shell/sbatch syntax, and the smoke
  dry-plan (`2` inputs, `2` designs/input, `4` expected designs, array
  `0-0%1`).
- Initial infrastructure error: sandboxed commands failed before execution with
  `bwrap: Creating new namespace failed ... (ENOSPC)`. Re-running the same
  non-mutating checks outside the nested namespace resolved this; it was not a
  pipeline or data failure.
- Blocking error: the documented test interpreter
  `/home/users/zhkim216/code/envs/uv/elix_local/bin/python` does not exist on
  the current host, so the two focused pytest files have not run.
- Available runtime evidence: `/scratch/users/zhkim216/envs/uv/elix/bin/python`
  exists, and the repo's Sherlock setup points to
  `/scratch/users/zhkim216/containers/elix.sif`; the scratch interpreter is
  known to require the compatible container boundary on this login node.
- Status: stopped before `sbatch`, pending approval to run the focused tests in
  the existing `elix.sif` runtime and then submit only if they pass.

### 2026-07-16: sampling producer and downstream `model_name` disagree

- Symptom: the sampling adapter writes
  `model_name=abacus-t_sstate_selfcond_650M_noise`, while backmapping,
  validation, and the Hydra config require the exact canonical value
  `abacus-t`.
- Impact: smoke sampling could successfully produce four designs, but the next
  global validation/backmapping gate would deterministically reject them.
- Cause: an adapter metadata label was made more specific than the established
  consumer contract. This is not an ABACUS-T model-internal bug.
- Proposed resolution: change only the denovoval sampling adapter metadata
  constant to `abacus-t` and add a regression assertion connecting the producer
  manifest to the backmapping/validation contract. Preserve the full inference
  variant in `task_contract.json`, where checkpoint, ESM, hashes, and
  hyperparameters are already recorded.
- Status: unresolved; stopped before `sbatch` for user confirmation.

### 2026-07-16: producer/consumer contract repaired and validation resumed

- Resolution: changed only the denovoval sampling adapter's manifest metadata
  label to canonical `abacus-t`. The exact self-conditioned 650M variant remains
  recorded by checkpoint/ESM/code hashes, hyperparameters, and the run
  fingerprint in `task_contract.json`; released ABACUS-T model code is unchanged.
- Regression: the repo test now loads the scratch sampling producer without
  invoking its CLI and requires its `MODEL_NAME` to equal the Hydra config's
  sole `sequence_design.model_names` value and canonical `abacus-t`.
- Environment resolution: focused tests run in `elix.sif` with
  `/scratch/users/zhkim216/envs/uv/elix/bin/python`; the missing home-side
  `elix_local` interpreter is not used.
- Fixed-position decision: keep `fixed_positions=""` and
  `designed_positions=""`. The nonempty file in the released `run.sh` is a
  4pn2-demo-specific mask and is invalid for 3,400 unrelated denovoval
  backbones; this override does not change the general self-conditioning
  hyperparameters.
- Status: implementation complete; post-change tests and smoke submission gate
  pending.

### 2026-07-16: smoke sampling job blocked by incompatible partition constraint

- Submission: job `34235779`, array `0-0%1`, for the two-input smoke manifest.
- Scheduler state: `PENDING (BadConstraints)` with no allocation or runtime;
  no preprocessing, sampling, or output artifact was created.
- Request: partitions `possu,bioe`, one GPU, and feature
  `GPU_SKU:A100_SXM4`.
- Diagnosis: the current `bioe` GPU nodes expose `GPU_SKU:A40`; the available
  A100-SXM4 node is in `possu`. The known containerized nativeval owner script
  requests `possu` alone for this A100 constraint.
- Proposed resolution: cancel the never-started job, change only the denovoval
  sampling sbatch partition to `possu`, rerun shell syntax and dry-plan checks,
  and submit a new smoke job with the same manifest and runtime hashes.
- Status: stopped for user confirmation before cancellation, launcher edit, or
  resubmission.

### 2026-07-16: sampling scheduler constraint repaired

- Cancelled: job `34235779` without allocation or runtime.
- Resolution: sampling arrays now request partitions `bioe,possu,owners` and
  the explicit OR constraint
  `GPU_SKU:A40|GPU_SKU:A100_SXM4|GPU_SKU:H100_SXM5|GPU_SKU:H200_SXM5`.
- Retry semantics: added `#SBATCH --requeue`; scheduler-level interruption may
  requeue the same job, while deterministic application errors still write a
  failed `status.json` and are not submitted repeatedly by a monitor.
- Validation: shell syntax and the two-input/four-design dry-plan passed;
  `sbatch --test-only` accepted the request and selected a valid A40 candidate
  in `bioe` rather than returning `BadConstraints`.
- Status: resolved; replacement smoke submission pending.

### 2026-07-16: replacement smoke sampling submitted

- Job: `34237253`, array `0-0%1`.
- Verified scheduler contract: `Partition=bioe,possu,owners`,
  `Features=GPU_SKU:A40|GPU_SKU:A100_SXM4|GPU_SKU:H100_SXM5|GPU_SKU:H200_SXM5`,
  `Requeue=1`, one GPU, eight CPUs, 24 GiB, four-hour limit.
- Scheduler state at submission audit: `PENDING (Priority)`, not a constraint
  error; scheduler estimated start `2026-07-16T16:44:16` PDT.
- Artifact state: no task runtime or output yet. The smoke sampling gate remains
  in progress and requires `status.json.state=complete` plus exactly four
  validated `design_manifest.csv` rows before backmapping.

### 2026-07-16: smoke sampling failed on an unsupported H100 allocation

- Job: `34237253_0` ran on `owners` node `sh04-02n01` and exited `1:0`
  after 3 minutes 47 seconds. The task status is `failed`; no design FASTA or
  PDB was accepted, and no downstream step was started.
- Preprocessing succeeded for both smoke inputs. The logs show both ligand SDFs
  loaded and both `A1L3W_len150_0.npy` and `NA_len150_0.npy` presented to the
  sampler.
- Hardware/runtime mismatch: the assigned node advertises
  `GPU_SKU:H100_SXM5` and compute capability `9.0`, while the released ABACUS-T
  PyTorch runtime reports support only through `sm_86` and warns that the H100
  is incompatible.
- Why the direct error was hidden: the released sampler wraps each input in
  `except RuntimeError`, prints only when the message contains `out of memory`,
  and otherwise continues. A CUDA kernel incompatibility can therefore skip
  both inputs, return process exit code zero, and leave the output group empty.
  The denovoval adapter correctly converted that silent skip into the explicit
  failure `expected ['A1L3W_len150_0', 'NA_len150_0'], got []`.
- Root-cause confidence: high. The node SKU, PyTorch architecture warning,
  original exception suppression, two processed-input log lines, and completely
  empty `designs/` directory agree. This is distinct from the validated output
  group name, which exactly matches the original writer format.
- Pending decision: either restrict this legacy runtime to compatible A40/A100
  GPUs (recommended), or replace/upgrade the runtime so H100/H200 kernels are
  supported. No ABACUS-T internal code has been changed, no retry has been
  submitted, and the smoke gate remains stopped for user confirmation.

### 2026-07-16: compatible-GPU retry policy approved

- User decision: restrict ABACUS-T sampling to A40 (`sm_86`) and A100-SXM4
  (`sm_80`); H100/H200 are no longer legal for this legacy Torch runtime.
- Launcher resolution: keep partitions `bioe,possu,owners`, one GPU, and
  `#SBATCH --requeue`, but narrow the feature constraint to
  `GPU_SKU:A40|GPU_SKU:A100_SXM4`.
- Adapter resolution: before preprocessing or invoking the released sampler,
  compare the allocated device capability with `torch.cuda.get_arch_list()` and
  execute a one-element CUDA kernel plus synchronize. Missing CUDA, unsupported
  architecture, or a failed kernel now raises an explicit task failure instead
  of reaching the upstream sampler's broad `RuntimeError` handler.
- Upstream boundary: no released ABACUS-T model or sampler code was changed.
- Retry gate: syntax, focused regression tests, dry plan, and scheduler
  `--test-only` must pass before guarded overwrite of only the failed smoke task
  and exact resubmission.

### 2026-07-16: compatible-GPU retry validation passed

- Passed: sampling sbatch `bash -n`, adapter compilation in the actual Python
  3.8 ABACUS-T sandbox, repo targeted checks, and the focused repo suite
  (`15 passed`).
- Contract matrix passed: A100 `sm_80` and A40 `sm_86` are accepted; H100/H200
  `sm_90` and missing CUDA are rejected. The installed Torch
  `1.10.1+cu111` build flags are exactly `sm_37, sm_50, sm_60, sm_70, sm_75,
  sm_80, sm_86`.
- Dry plan passed: the retry selects two smoke inputs, two designs per input,
  exactly four expected designs, array `0-0%1`, and `OVERWRITE=1`.
- Scheduler validation passed: `sbatch --test-only` accepted the narrowed
  constraint and selected `bioe` node `sh03-16n13`, which advertises A40 and
  compute capability `8.6`.
- Independent reviews found no blocking issue. They confirmed the failed task
  is terminal, no duplicate ABACUS-T job or downstream artifact exists, and the
  guarded deletion scope resolves only to
  `sampling/smoke/tasks/task_0000`.
- Validation-command corrections: the first compile command named a nonexistent
  `.sif`; the owner script actually defaults to the existing `.sandbox`, where
  compilation passed. A later login-node diagnostic saw an empty
  `torch.cuda.get_arch_list()` because no GPU was visible; build flags were
  instead read from Torch compile metadata. The real allocation-dependent CUDA
  allocation/add/synchronize probe remains a required live-smoke check.
- Status: all pre-submit gates passed; exact guarded smoke retry authorized.

### 2026-07-16: compatible-GPU smoke retry submitted

- Job: `34248513`, array `0-0%1`, with `OVERWRITE=1` for the exact failed
  `sampling/smoke/tasks/task_0000` path.
- Verified request: partitions `bioe,possu,owners`, feature constraint
  `GPU_SKU:A40|GPU_SKU:A100_SXM4`, one GPU, eight CPUs, 24 GiB, four hours,
  and `Requeue=1`.
- Initial scheduler state: `PENDING` with no failure reason. Completion still
  requires the live CUDA preflight, terminal scheduler success,
  `status.json.state=complete`, and exactly four validated manifest rows.

### 2026-07-16: smoke retry stopped on a Python 3.8 overwrite-guard error

- Job: `34248513_0` started on `owners` node `sh03-11n01` under the approved
  A40/A100 constraint, then failed `1:0` after 15 seconds. No downstream step
  was started.
- Direct error: `guarded_remove_task()` called
  `task_dir.name.removeprefix("task_")`; `str.removeprefix()` is unavailable in
  the actual ABACUS-T Python `3.8.12` runtime and raised `AttributeError`.
- Mutation impact: the exception occurred before `shutil.rmtree()`. The old
  failed `task_0000`, staging manifest/PDB/SDF files, top-level logs, and all
  downstream paths remain unchanged. Slurm reports no restart or active retry.
- Why pre-submit validation missed it: Python 3.8 compilation proves syntax but
  not runtime availability of a string method, the focused repo tests ran in a
  newer Python, and no test invoked the overwrite guard through the actual
  Python 3.8 runtime.
- Compatibility scan: this is the only `removeprefix`/`removesuffix` call in the
  denovoval task runner. Postponed annotations make its `list[...]`, `dict[...]`,
  and `X | None` annotations safe in Python 3.8; those annotations already ran
  successfully through manifest planning and job startup.
- Proposed minimal resolution: retain the exact path/symlink guard but parse
  the already-regex-constrained `task_NNNN` name with a Python 3.8-compatible
  slice or dedicated pure helper. Add both a unit test for valid/invalid task
  names and an actual Python 3.8 container smoke of that helper before retry.
- Status: unresolved; stopped before editing or resubmitting for user
  confirmation, as required by the anomaly gate.

### 2026-07-16: overwrite guard made Python 3.8-compatible

- User decision: keep the released ABACUS-T Python/Torch/CUDA environment
  unchanged and repair only the denovoval adapter.
- Resolution: `guarded_remove_task()` now requires the existing exact
  `task_NNNN` regex, parses the numeric suffix with a Python 3.8-compatible
  slice, and then retains the existing mode-range, resolved-path, and symlink
  guards before `shutil.rmtree()`.
- Regression scope: exact canonical task removal must preserve an outside
  sentinel; invalid names, out-of-range IDs, and symlinks must be rejected
  without deletion. The positive destructive smoke must also run under the
  actual ABACUS-T Python 3.8 sandbox, not only the newer repo test runtime.
- Environment/upstream boundary: no environment, checkpoint, ESM, released
  preprocessor, or released sampler file was changed.
- Status: code and focused regression added; validation and smoke resubmission
  pending.

### 2026-07-16: Python 3.8 cleanup regression and critic review passed

- Focused validation passed: repo targeted checks, the combined ABACUS-T suite
  (`16 passed`), and source execution under the actual ABACUS-T Python 3.8
  sandbox.
- Actual-runtime destructive safety passed in temporary scratch roots only:
  exact `task_0000` removal preserved an outside sentinel, while invalid names,
  out-of-range task IDs, and a leaf symlink were rejected without deleting
  their directories or targets.
- Test-safety correction: `runpy.run_path()` returns a mapping distinct from
  the loaded functions' `__globals__`. This was detected before running any
  cleanup regression; tests now redirect the shared function globals to
  `tmp_path` explicitly and assert both guard functions share that namespace.
- Static compatibility scan found no remaining `removeprefix` or
  `removesuffix` call in the denovoval task runner.
- Post-fix critic result: `GO`, with no blocking finding. Current real task and
  all parent directories are ordinary non-symlink directories.
- Non-blocking residual hardening: `strict_task_path()` resolves the `tasks/`
  root itself, so a future parent-root symlink is not separately rejected.
  The current canonical tree is not symlinked, and exact leaf/path/range guards
  make the approved smoke retry safe; broader parent-root hardening is deferred.
- Status: compatibility fix validated; exact smoke retry may proceed after the
  dry-plan and scheduler test gates.

### 2026-07-16: compatible-GPU smoke retry submitted

- Pre-submit gates passed: sbatch shell syntax, exact smoke dry-plan, Slurm
  `--test-only`, absence of an active duplicate ABACUS-T job, terminal state of
  failed jobs `34237253` and `34248513`, and non-symlink ancestry for the exact
  overwrite target `sampling/smoke/tasks/task_0000`.
- Dry-plan contract: array `0-0%1`, 2 inputs, 2 designs per input, 4 expected
  designs, manifest SHA256
  `20dd4b2f2e340109dc0bf65cf957e29fc9eab630c0598525037434b82f183028`.
- Submitted job `34255390` with `MODE=smoke`, `OVERWRITE=1`, partitions
  `bioe,possu,owners`, constraint `GPU_SKU:A40|GPU_SKU:A100_SXM4`, and
  `--requeue`.
- Runtime digests exported at submission: checkpoint
  `4b68020d9eb0ac0f3bab96255a7060ac8185f6808421de61c535dc4d18d8cc85`,
  ESM
  `ea9d0522b335a8778dea6535a65301f10208dece28cd5865482b0b1fc446168c`,
  original sampler
  `0e6a1417a2b42b0167267794eabecb36ddd6054dd4f4bc85ba753a6fbbd822f6`.
- Status: submitted; scheduler, preflight, sampling, and artifact gates pending.

### 2026-07-16: scheduler availability diagnostic quoting error

- A read-only `sinfo` diagnostic initially passed an unquoted output format
  containing shell pipe characters. Bash interpreted the format segments as
  commands and returned `command not found`; this did not touch the submitted
  job or any artifact.
- Fix: quote the complete `sinfo -o` format, then filter the resulting node
  records explicitly for `GPU_SKU:A40` and `GPU_SKU:A100_SXM4`.
- Corrected evidence: eligible nodes exist in the approved partitions, but the
  observed matching inventory was `mix`, `mix-`, `alloc`, or `drain*`,
  with no matching `idle` node. Job `34255390_0` remaining `PENDING` is
  therefore consistent with compatible-GPU resource waiting, not a malformed
  feature request.
- Status: no pipeline state changed; monitoring continues.

### 2026-07-16: AF3 smoke launcher dry-check sandbox retry

- Initial `bash -n` and two dry command-render checks were blocked before
  launcher execution by the recurring workspace sandbox error
  `bwrap: Creating new namespace failed ... ENOSPC`.
- Fix: rerun the same read-only checks outside the exhausted namespace sandbox;
  no code, job, or artifact state was changed.
- All checks then passed. The smoke input-generation and inference actions both
  resolve to `--num-arrays 1 --num-recycles 1 --num-diffusion-samples 1
  --smoke --array-id 0`; only their final action flags differ.
- Status: the approved smoke AF3 index/sample contract is represented correctly;
  actual downstream submission remains gated on sampling and backmapping.

### 2026-07-16: smoke retry bounded queue monitor

- At 17:59 PDT, job `34255390_0` remained `PENDING`, elapsed `0:00`,
  node unassigned, reason `None`, and `Restarts=0`. Slurm estimated start
  time was 19:06:09 PDT.
- The job-specific stdout and stderr files did not yet exist, proving the batch
  script had not entered execution. Guarded overwrite, CUDA preflight,
  preprocessing, and sampling therefore had not run.
- `task_0000/status.json` remained the 277-byte prior H100-failure artifact
  with mtime 15:24:28, and no `design_manifest.csv` existed. Existing task
  and staging artifacts were not mutated.
- Scheduler and artifact state agree: there is no new failure or wrong-GPU
  allocation, only compatible-GPU queue wait. Metric and downstream state remain
  absent.
- Status: smoke retry remains active in Slurm; resume monitoring at allocation
  or after the scheduler estimate, before any downstream action.

### 2026-07-16: A100 smoke sampling failed after successful preflight

- Job `34255390_0` started at 18:14:19 PDT on `sh03-18n05` in `possu`
  with one allocated GPU and ended `FAILED 1:0` at 18:16:52
  (`Restarts=0`).
- The adapter safely replaced the old failed task, printed a new fingerprint
  `6a033a82e9f9f0d787f8c2edb70eff123bc999a812676b7161aa9876d9730856`,
  and passed its live tensor preflight on `NVIDIA A100-SXM4-80GB`
  (`sm_80`).
- Preprocessing then completed for both `A1L3W_len150_0` and
  `NA_len150_0`. Both NPY files and copied ligand SDF/PDB files exist, and the
  adapter's exact ligand-graph validation completed before sampling.
- The released sampler loaded
  `esm2_t33_650M_UR50D.pt` and
  `checkpoint_best_noise650M.pt`, then failed at its first `model.cuda()`
  with `RuntimeError: CUDA error: all CUDA-capable devices are busy or
  unavailable`.
- No design files or `design_manifest.csv` were produced. `status.json`
  correctly records `state=failed`; downstream backmapping and AF3 were not
  started.

#### Diagnosis

- Strongest root cause: the custom adapter's preflight creates a CUDA context
  in the long-lived parent process using a live CUDA tensor and
  `torch.cuda.synchronize()`. The adapter subsequently launches the released
  sampler as a child process while the parent context remains alive. On a GPU
  using exclusive-process semantics, the child cannot acquire the device and
  fails at `model.cuda()`.
- Supporting evidence: Slurm allocated one GPU; the parent live tensor operation
  succeeded; the child inherits the same `CUDA_VISIBLE_DEVICES`; failure occurs
  before sampling and after both model files load. This rules out an absent or
  incompatible GPU and makes an input/CCD/preprocessing fault unrelated to the
  observed failure.
- Lower-confidence alternative: a transient node-level GPU ownership problem.
  A clean retry on another allocation could distinguish it, but retrying the
  unchanged parent-context pattern risks the same deterministic failure.
- Recommended adapter-only repair: execute the same live tensor preflight in a
  short-lived subprocess and require its success before preprocessing. The
  subprocess exits and releases its CUDA context before the released sampler
  starts. Do not change the ABACUS-T environment or released sampler.
- Status: anomaly gate is closed. No code fix, deletion, retry, backmapping, or
  AF3 submission will occur until the user approves this repair direction.

### 2026-07-16: child-process preflight design critic review

- Read-only critic result: `GO after user approval`; no file or scheduler
  mutation was performed.
- Required implementation boundary: expose `cuda-preflight` as a standalone
  task-runner CLI branch that requires no mode or manifest. Invoke it as
  `[sys.executable, "-B", resolved_task_runner, "cuda-preflight"]` with an
  explicit stable working directory, copied environment, inherited
  stdout/stderr, and a required zero exit status.
- `validate_cuda_runtime()` must only run inside that child branch.
  `execute()` must wait for the child to exit before preprocessing and must
  not import or call Torch/CUDA in the parent.
- The task-runner SHA in `task_contract.json` will intentionally change after
  this adapter edit; checkpoint, ESM, released preprocessor, and released
  sampler digests remain unchanged.
- Required regression matrix: exact child command/environment/cwd; successful
  child exits before preprocessing; nonzero child prevents preprocessing and
  sampling and produces failed task status; CLI calls the live validator once
  and exits; existing supported/unsupported architecture contract cases remain
  passing; final proof is a live compatible-GPU smoke retry.
- Residual uncertainty: a transient node/driver fault could still recur after
  context teardown. If it does, stop at the next anomaly instead of attributing
  it to the adapter.
- Status: repair design is decision-complete but remains unimplemented pending
  explicit user approval.

### 2026-07-16: same-node current GPU state rules against an unchanged retry

- At the user's request, the failed node was checked directly without another
  broad validation pass. The active shell is on `sh03-18n05.int`, the same
  node used by job `34255390`.
- Current `nvidia-smi` evidence for the visible A100-SXM4-80GB: compute mode
  `Exclusive_Process`, 1 MiB used, 0% utilization, and no compute process.
- The user reported that another workload was running around the original
  failure. That can explain historical contention in general, but it does not
  explain this specific sequence: the adapter's live tensor preflight itself
  succeeded first, so that long-lived parent acquired the only CUDA context
  allowed by `Exclusive_Process`; the released sampler then failed as a
  separate process at `model.cuda()`.
- The task-runner and released-sampler hashes remain exactly those from the
  failed run. There is no active ABACUS-T duplicate, and job `34255390` remains
  terminal `FAILED 1:0`.
- Decision: do not submit an unchanged retry that is expected to repeat on an
  A100 in exclusive-process mode. The adapter-only child-preflight repair
  remains the minimal corrective boundary.
- Status: no code, task artifact, or scheduler state was changed by this check.
### 2026-07-16: pipeline reset to preparation-only ownership

- User decision: discard the denovoval adapter stack that combined preparation,
  CUDA preflight, ESM/checkpoint loading, sampling, and task cleanup. The
  current scope stops after the released preprocessor produces NPY files and
  adapter-owned validate_npy accepts them.
- Retired integration files:
  run_denovoval_sstate_task.py, run_denovoval_sstate_task.sh,
  run_denovoval_sstate_array.sbatch, submit_denovoval_sstate.sh, and
  preprocess_denovoval_sstate_sdf_first.py.
- Replacement:
  /scratch/users/zhkim216/code/ABACUST-v2-pub-main1/scripts_jinho/prepare_denovoval_sstate.py.
  It exposes only prepare-chunk and explicit finalize; it never loads CUDA,
  ESM weights, a sampling checkpoint, or the released sampler.
- Frozen input boundary: the staging manifest plus its staged PDB/SDF files are
  the preparation source of truth. Source CIF and live CCD mirror files are not
  re-required. The released ABACUS-T preprocessor remains byte-for-byte
  unchanged and is invoked with SDF-first ligand-file priority.
- Execution contract: smoke is one 2-input chunk, per-CCD is one 154-input
  chunk, and full is ten deterministic 340-input chunks. Chunk results are
  immutable and hash-reused; incompatible completed results hard-fail. Failed
  attempts are retained outside the publishable tree. finalize publishes a
  gate directory only after exact chunk/sample/hash reconciliation.

### 2026-07-16: preparation implementation and test-environment errors

- Edit infrastructure error: the required apply_patch helper intermittently
  failed before reading files with "bwrap: Creating new namespace failed ...
  (ENOSPC)", while many independent Codex sessions were active on the node.
  No target file was changed by those failed attempts. The authorized diff was
  applied with the standard patch utility outside the exhausted nested
  namespace; exact patch backup/reject artifacts were removed afterward.
- Test-runtime error 1: the documented
  /home/users/zhkim216/code/envs/uv/elix_local/bin/python does not exist on
  this node.
- Test-runtime error 2: invoking
  /scratch/users/zhkim216/envs/uv/elix/bin/python directly on the host failed
  before pytest with "GLIBC_2.27 not found"; this environment requires its
  compatible container boundary. Host and ABACUS-T environments also do not
  install pytest.
- Validation resolution: both touched test files compile, the new entrypoint
  compiles in the actual ABACUS-T Python 3.8 environment, and equivalent pure
  manifest/chunk/PDB-sequence/bond-endpoint/bond-class/zero-bond assertions
  passed there. The actual two-input preparation smoke below provides the
  consumer-path validation that unit-only execution cannot.

### 2026-07-16: two-input preparation smoke complete

- Published path:
  /scratch/users/zhkim216/out_dir/benchmarks/denovoval/abacus-t/preparation/smoke.
- Inputs: exactly A1L3W_len150_0 and NA_len150_0; staging-manifest SHA256
  20dd4b2f2e340109dc0bf65cf957e29fc9eab630c0598525037434b82f183028.
- A1L3W: 150-residue chain-A protein, one SDF-first ligand, 26 atoms, 28
  bonds, exact graph endpoints/classes, coordinates, atom/charge features, and
  UniMol atom count.
- NA: 150-residue chain-A protein, one SDF-first ligand, one atom, zero
  bonds, exact empty edge shapes, coordinates, atom/charge features, and
  UniMol atom count.
- Publication status is complete; all missing/surplus/failed/hash diagnostics
  are zero. Preparation-manifest SHA256 is
  8e318e5bd0724aea8e848f45066d2a3ef03d1f352dde21a6a491d0a2030def4e.
  A repeated prepare-chunk call hash-validated and reused the published gate
  without rerunning preprocessing.
- Status: preparation smoke gate passed. Per-CCD and full preparation have not
  been run in this scope.
### 2026-07-16: workspace-race critic fix and corrected smoke

- Independent code review found one blocking race: two externally parallel
  chunks could both observe a missing workspace, or one could observe the
  directory before run_contract.json existed.
- Resolution: a mode-scoped fcntl file lock now serializes only workspace
  initialization and contract comparison. A synchronized two-process
  regression passed in the actual ABACUS-T Python 3.8 environment.
- Additional narrow hardening rejects "." and ".." sample/basename path tokens
  and records run fingerprint, entrypoint SHA, and released-preprocessor SHA in
  every preparation-manifest row.
- Because those edits changed the entrypoint hash, the old complete smoke was
  correctly rejected as contract-incompatible rather than overwritten. It was
  preserved under preparation/superseded_runs/smoke/18f79d69... and the smoke
  was regenerated.
- Current smoke fingerprint:
  8911aa08bf185bcb9cdb947ac14f01a747e5e61bb181e8658277907d2ad71677.
  Current preparation-manifest SHA256:
  aa9b749229415cd0f7e0284c30bebbe4ae7074d32b17bd62bb75f6f0527f32f5.
- Independent re-review decision: GO; no remaining blocking implementation
  issue before the per-CCD gate.

### 2026-07-16: per-CCD preparation stopped on raw-versus-sanitized graph comparison

- Gate: one length-150 input for each of 154 CCDs. The released preprocessor
  completed its CPU pass, but adapter validation stopped at the first manifest
  row, 12C_len150_0. Full preparation was not started.
- Preserved failure:
  preparation/failed_attempts/per_ccd/99504e210a7d4b47/
  chunk_0000.20260716T224036.65159.
- Direct error: "12C_len150_0: NPY ligand graph differs from prepared SDF".
- Diagnosis: this is a validator representation-boundary bug, not an ABACUS-T
  preprocessing or staging chemistry failure. The validator re-read the SDF
  with sanitize=False/removeHs=False, producing zero aromatic atoms and graph
  SHA 250d3bbb.... The released preprocessor reads with
  sanitize=True/remove_hs=True; its serialized NPY ligand has 19 aromatic atoms
  and graph SHA 80fd5068..., exactly equal to both a sanitized SDF re-read and
  the frozen staging-manifest graph SHA.
- Proposed smallest fix: change only the adapter validation re-read to
  sanitize=True/removeHs=True, matching the released consumer boundary. Keep
  the exact serialized-NPY versus manifest hash, PyG edge/class, coordinate,
  atom/charge, and UniMol checks unchanged.
- Status: stopped for user confirmation before editing or retrying the
  per-CCD gate.

### 2026-07-17: sanitized graph validation fix and preparation completion

- Applied the approved validation-only fix in
  `prepare_denovoval_sstate.py`: the adapter now re-reads each prepared SDF
  with `sanitize=True, removeHs=True`, exactly matching the unchanged released
  ABACUS-T preprocessor. No upstream preprocessing or model code was changed.
- Regenerated/finalized all three immutable preparation gates successfully:
  smoke has 2 samples, per-CCD has 154 samples, and full has 3400 samples in
  ten 340-sample chunks. Every `status.json` reports `state=complete`.
- Current preparation-manifest SHA256 values are
  `fa555465fac5db857c6d3b78b61cfb479fe2044700244d87044228c41a81ede1`
  (smoke),
  `b3c902f5e7860818f5b78c1c5e5ea9b2888c2c2df26cf3e1ebbb910ad00605b2`
  (per-CCD), and
  `f924e639ccf4c9de12375271b53d55cb8a2ec243414e3676f2ec0cfb45370be5`
  (full).

### 2026-07-17: prepared-NPY sampling launcher and container boundary

- Added `run_denovoval_sstate_sampling.sh`. It consumes only a completed
  prepared-NPY chunk, never reruns preprocessing, and calls the unchanged
  released sampler with the `model/abacusT_sstate/run.sh` inference contract:
  batch size 2, temperature 0.1, PLM self-conditioning enabled with ESM-2
  650M, 20 iterations, noise650M checkpoint, augment epsilon 0.2, and no
  denovoval fixed/designed-position override.
- Initial shell compatibility error: the available Bash does not provide the
  newer array-reading form first used by the launcher. Replaced it with the
  Bash-compatible `mapfile -t` form. The launcher also checks the exact smoke
  IDs and atomically reserves a missing output leaf before the released
  sampler starts, preventing silent overwrite and concurrent duplicate runs.
- Initial host smoke failed before CUDA/model execution while importing
  fairseq through lxml: the host runtime lacked `GLIBC_2.28`. The empty reserved
  leaf was verified to contain zero entries and preserved, not deleted, under
  `sampling/failed_attempts/smoke/20260717_063511_host_glibc/`.
- Added two small, separate runtime helpers that reuse
  `abacust_container_env.sh`: `shell_in_container_abacus.sh` opens an
  interactive ABACUS-T shell, while `wrap_sbatch_in_container_abacus.sh`
  copies and submits an arbitrary sbatch script through the same container.
  They do not alter the ABACUS-T environment or upstream sampler.
- Interactive-shell error: the host inherited a VS Code `PROMPT_COMMAND`, so
  the clean container shell printed `__vsc_prompt_cmd_original: command not
  found` on every prompt. Unsetting only `PROMPT_COMMAND` immediately before
  `exec bash --noprofile --norc -i` removed the prompt error.

### 2026-07-17: two-sample ABACUS-T sampling smoke complete

- Ran the smoke chunk in the validated ABACUS-T container on device 0. Runtime
  logs confirmed loading `esm2_t33_650M_UR50D.pt` and
  `checkpoint_best_noise650M.pt`.
- Both exact smoke members completed: `A1L3W_len150_0` and `NA_len150_0`.
  Output is under
  `/scratch/users/zhkim216/out_dir/benchmarks/denovoval/abacus-t/sampling/smoke/chunks/chunk_0000/T_0.1_R_20_selfcond_P3_0.2`.
- Exact inventory validation passed: 2 sample directories, 10 total files,
  4 final-iteration design PDBs, 2 three-record design `.fa` files, 2 ligand
  SDFs, and 2 base/native PDBs.
- The released sampler reported `violation=1` for
  `NA_len150_0_design_0_19`; the other three designs reported zero. This is the
  sampler's per-design count of newly introduced residue-level structural
  violations relative to the input, not a runtime or inventory failure. It is
  recorded here without changing or filtering the upstream result.
- Audit-command correction: the first independent count used `.fasta` and
  `*_native.pdb` patterns, but the released sampler writes `_design.fa` and
  `<sample_id>.pdb`. That produced false zero counts only in the audit command;
  rerunning with the released output names returned the exact inventory above.

### 2026-07-17: chunks-only sampled-output manifest gate complete

- Retired the failed legacy task-layout artifact from the active smoke tree.
  Nothing was deleted: the 12-file `sampling/smoke/tasks` tree was moved to
  `sampling/failed_attempts/legacy_task_adapter/20260716_cuda_busy/tasks`.
- Added the Elix-owned `index_sampled_designs.py` boundary. It supports smoke,
  per-CCD, and full preparation layouts, but this gate executed smoke only. It
  reads the completed preparation manifest/status, the matching frozen staging
  manifest, and the released sampler's `chunks` output; ABACUS-T code and its
  environment remain unchanged.
- The task-based manifest glob contract was removed. Config, the global
  sampling validator, and the backmapper loader now consume one atomic
  `sampling/<mode>/design_manifest.csv`. Backmapping CIF generation and AF3
  were not run in this gate.
- Host test-runtime boundary: the shell-default Python is 3.8.16 and does not
  install pytest. Focused tests were therefore run through
  `shell_in_container_elix.sh` using the Elix Python 3.12 environment.
- Focused-test fixture error: the first run had 6 passing tests and one failure
  because the new test called `mkdir()` on pytest's already-existing
  `tmp_path`. Removing that redundant call made the rerun pass.
- Critic review found one blocking error-path gap: missing or malformed
  preparation/staging inputs could raise before the index validation report was
  written. The indexer now converts such source-boundary exceptions into an
  `index_input_error`, persists a failed report, and refuses manifest
  publication. A focused regression proves this behavior.
- Final focused result: 8 tests passed. Targeted Python compile and YAML syntax
  checks also passed.
- Published smoke manifest:
  `/scratch/users/zhkim216/out_dir/benchmarks/denovoval/abacus-t/sampling/smoke/design_manifest.csv`.
  It contains exactly four rows: two source samples times design indices 0 and
  1 at iteration 19. Manifest SHA256 is
  `362430b37c4ce6337e4f45db06fb98d97ac3dc5a74472a3d5cc15cb9744a3da1`.
- Producer index validation and the independent global sampling consumer both
  report `status=complete`, expected/observed rows 4, and total diagnostics 0.
  The backmapper's new single-manifest loader read all four rows. A repeated
  index run reused the byte-identical manifest with no temporary files.
- Status: smoke sampled-output manifest gate passed. Per-CCD/full manifests,
  semantic CIF backmapping, AF3 JSON, AF3 inference, and metrics remain
  unexecuted.

### 2026-07-17: bundled smoke backmapping, AF3, and metrics complete

- Ran the smoke downstream work as one macro-gate: semantic CIF backmapping,
  AF3 JSON generation and schema parsing, one AF3 prediction per design, and
  self-consistency/docking metrics including TM-align. Successful substeps did
  not require separate user pauses; actual anomalies stopped downstream work.
- Runtime-boundary error: directly invoking
  `/scratch/users/zhkim216/envs/uv/elix/bin/python` on the host failed while
  importing NumPy because the host `/lib64/libm.so.6` lacked `GLIBC_2.27`.
  The import failed before writing any backmapping output. Re-running the same
  CPU command through the existing `shell_in_container_elix.sh` boundary fixed
  the runtime mismatch; no environment or dependency was modified.
- Backmapping-contract error: all four designs were initially rejected because
  `validate_sampled_protein()` incorrectly required the ABACUS-T sampled
  backbone coordinates to equal the canonical source backbone within
  0.00051 A. Boundary measurements localized the change: canonical CIF to
  staged PDB was at most 0.000500 A, staged PDB to prepared PDB was exactly
  0 A, and prepared PDB to sampled PDB was up to 0.116 A. Thus ABACUS-T itself
  had produced a displaced sampled backbone; staging had not changed it.
- Approved resolution: `backmap_designs.py` now continues to require the exact
  protein chain, sequence, residue identifiers, N/CA/C/O topology, finite
  coordinates, and protein-only records, but records source-to-sampled
  backbone displacement instead of rejecting it. The sampled protein is
  preserved in the semantic CIF and is still checked field-for-field by the
  sampled-PDB-to-CIF roundtrip validator. The ligand remains copied from the
  canonical source CIF and is checked for exact CCD, atom annotations,
  coordinates, occupancy, B factor, and charge. No upstream ABACUS-T code was
  changed.
- Regression coverage adds a deliberately displaced sampled backbone that must
  survive the CIF roundtrip and a residue-identity mismatch that must still
  fail. Targeted compilation passed and
  `test_denovoval_abacust_backmap_af3.py` passed all 9 tests.
- Backmapping completed for all four rows with failed/missing rows 0. The
  independent global backmapping validator reported diagnostics 0. The
  backmapped manifest is
  `/scratch/users/zhkim216/out_dir/benchmarks/denovoval/abacus-t/backmapped/smoke/design_manifest.csv`
  with SHA256
  `b8530c43a4530ea4d7ef075de82b914a675b6fc2c5902b56ddd5fee4b93013dd`.
  Original CCDs were restored as `A1L3W` and `NA`; sampled-protein and
  canonical-ligand CIF coordinate roundtrip errors were 0 for every row.
- Submitted AF3 smoke job `34329345` through
  `wrap_sbatch_in_container_elix.sh` with array `0-0`, `RUN_MODE=smoke`,
  `ACTION=run-af3`, and `ELIX_REQUEUE_ON_USR2=1`. The job ran on
  `sh04-10n01` and completed in 1:23 with scheduler state `COMPLETED` and exit
  code `0:0`; no retry or requeue was needed.
- The AF3 stderr reported one empty Tokamax autotuning-cache JSON as a parse
  error, then explicitly treated it as a cache miss and continued autotuning.
  This did not propagate to the runner status, prediction inventory, or metric
  diagnostics and required no code or environment change.
- Validation-command error: after the completed `run-af3` action, the
  `af3-inputs` stage was mistakenly run even though that stage is specifically
  for `generate-inputs-only` status `input_ready`. It rejected the valid
  `complete` status. No AF3 artifact was invalid and no inference rerun was
  performed. The spurious `validation_inputs.json` was removed, and the
  correct `af3-complete` validator was run; that validator includes the same
  exact JSON semantic checks while requiring the completed-run status.
- Final smoke state: four AF3 JSONs have protein chain `A`, ligand chain `L`,
  the designed protein sequence, and original ligand CCD `A1L3W` or `NA`;
  four prediction rows, four self-consistency rows, and four docking rows are
  present. TM-align metrics are finite for all four rows. The final report at
  `/scratch/users/zhkim216/out_dir/benchmarks/denovoval/abacus-t/af3_ss/smoke/reports/validation_complete.json`
  reports `status=complete` and `total_errors=0`.

### 2026-07-17: one length-150 input per CCD sampling complete

- The prior failed `run_denovoval_sstate_array.sbatch` had been removed, so no
  scheduler owner remained for the validated direct sampler. Recreated one
  thin sbatch in `ABACUST-v2-pub-main1/scripts_jinho`. It only checks the
  smoke/per-CCD/full array cardinality and calls the existing
  `run_denovoval_sstate_sampling.sh`; it does not preprocess inputs or wrap the
  released model entrypoint again.
- Scheduler resources follow the approved contract: partitions
  `bioe,possu,owners`, A40/A100/H100/H200 GPU feature variants, one GPU, eight
  CPUs, 16 GB memory, 24-hour walltime, and scheduler requeue enabled.
- Critic review found that `#SBATCH --requeue` alone was ineffective: a
  restarted task would encounter its partial output leaf and the direct
  sampler would correctly refuse overwrite. Fixed this only in the local
  launcher. It now writes `.sampling_complete` after exact final inventory
  validation. A Slurm restart reuses a matching completed leaf; an incomplete
  leaf is preserved under `sampling/failed_attempts/<mode>/requeue/` before a
  clean restart. Initial/manual duplicate submissions still refuse overwrite.
  Dry-run fixtures proved both partial-output preservation routing and
  completed-output reuse; the temporary fixture tree was removed.
- `bash -n`, the exact 154-input per-CCD dry-run, the wrong-array rejection,
  retry dry-runs, and Slurm `--test-only` all passed before submission.
- Submitted array job `34330819` through
  `wrap_sbatch_in_container_abacus.sh` with `MODE=per_ccd` and array `0-0%1`.
  It ran on `sh03-16n12` in `bioe`, loaded
  `esm2_t33_650M_UR50D.pt` and `checkpoint_best_noise650M.pt`, and completed in
  7:18 with scheduler state `COMPLETED`, exit code `0:0`, and no requeue.
- Released-launcher inventory validation found exactly 154 sample directories,
  308 final-iteration design PDBs, 154 three-record FASTAs, 154 ligand SDFs,
  and 154 native/base PDBs. The completion marker records mode `per_ccd`, chunk
  0, 154 samples, and two designs per sample.
- `validate_outputs.py sampling` intentionally has only downstream smoke/full
  modes. The per-CCD-specific `index_sampled_designs.py --mode per_ccd` therefore
  remained the gate instead of expanding that unrelated CLI. It published
  exactly 308 manifest rows for 154 source samples and design indices 0 and 1,
  with missing/extra/duplicate/stale diagnostics 0.
- Published manifest:
  `/scratch/users/zhkim216/out_dir/benchmarks/denovoval/abacus-t/sampling/per_ccd/design_manifest.csv`.
  SHA256:
  `84d9539dc199fcaeb7a8ed63290b31d9214168d0f84911a0ecff505852f8fc24`.
  Its `index_validation.json` reports `status=complete` and `total_errors=0`.

### 2026-07-17: full ABACUS-T sampling complete after GPU compatibility retry

- The first full array, job `34331198`, was submitted with the previously
  approved A40/A100/H100/H200 feature set. Task 4 landed on an H100 node and
  emitted PyTorch's explicit incompatibility warning: the installed build
  supports CUDA architectures through `sm_86`, while H100 is `sm_90`.
  The released sampler then visited all 340 inputs without producing sample
  directories, and the local final-inventory guard failed on the first missing
  output. This was a runtime compatibility failure, not a ligand or prepared-NPY
  failure.
- The anomaly gate stopped the run. Job `34331198` was cancelled before any
  downstream indexing; its running, pending, and configuring tasks became
  cancelled, while task 4 remained failed. The incomplete full tree contained
  462 directories and 2,252 files and was removed only after `squeue` confirmed
  no task from that job remained active.
- Per user approval, changed only the scheduler owner: partitions are now
  `possu,bioe,owners`, and the allowed features are
  `GPU_SKU:A40|GPU_SKU:A100_SXM4|GPU_SKU:A100_PCIE`. H100 and H200 are excluded.
  The container, ABACUS-T environment, local direct sampler, released upstream
  sampler, checkpoint, and inference settings were not changed.
- `bash -n`, an exact full task-0 container dry-run, and Slurm `--test-only`
  passed. The diff against the first submission's frozen owner copy contained
  only the partition-order and GPU-constraint lines.
- Resubmitted full job `34331994` with `MODE=full` and array `0-9%10`. All ten
  tasks ran on A40 nodes, loaded `esm2_t33_650M_UR50D.pt` and
  `checkpoint_best_noise650M.pt`, and finished `COMPLETED` with exit code `0:0`.
  Per-task elapsed times were 13:28 to 15:08; no requeue was needed.
- Aggregate sampling validation found exactly 3,400 sample directories, 6,800
  final-iteration design PDBs, 3,400 FASTAs, 3,400 ligand SDFs, 3,400
  native/base PDBs, and 10 completion markers: 17,010 files total. All ten
  markers record the correct chunk index, 340 samples, and two designs per
  sample. Current-job logs contain ten completion lines and no incompatible-GPU,
  traceback, OOM, or launcher-error pattern.
- Invocation note: `shell_in_container_elix.sh` opens an interactive shell and
  did not execute commands piped into its stdin; that attempt wrote no artifact.
  The existing `elix.sif` was then used with direct `singularity exec`. The first
  direct index command remained active while hashing the 6,800 designs; a
  premature filesystem check led to one repeated index call. The repeated call
  detected and reused the byte-identical manifest, so there was no duplicate or
  inconsistent publication.
- Full sampling indexing and independent validation are complete with 6,800 of
  6,800 design rows and diagnostics 0. The manifest is
  `/scratch/users/zhkim216/out_dir/benchmarks/denovoval/abacus-t/sampling/full/design_manifest.csv`
  with SHA256
  `48f2ce760cbe62419a3d9836076a938a1e1b68fc59993c4d4005446db4801ce7`.
