# Elix Working Agreement

## Repository

- Main package: `allatom_design/`; configs: `allatom_design/configs/` and
  `allatom_design/configs_local/`; operational scripts: `scripts/`.
- Use `rg`/`rg --files` for discovery. Inspect the actual caller, config, log,
  or artifact before making a code-grounded claim.
- Sherlock Python belongs inside the Torch 2.8 container with
  `/scratch/users/$USER/envs/uv/elix-torch280` activated. Host Python is not a
  valid runtime check for NumPy/Torch.
- Keep large outputs, checkpoints, caches, credentials, and private data out of
  git.

## Work

- Follow the latest user request for scope. A correction replaces the
  conflicting earlier assumption, but does not implicitly waive the
  external-state readiness gates below.
- Questions, explanations, reviews, and diagnoses are read-only unless the user
  also asks for a change. Build/fix requests authorize the direct, in-scope
  edits and checks needed to finish them.
- Ask only when a missing choice can materially change correctness, scope,
  destructive impact, or expensive external work. Otherwise inspect first,
  make the smallest reasonable assumption, and proceed.
- Use a plan only when it helps coordinate a genuinely multi-step or risky
  task. Do not turn a direct task into a planning or review workflow.
- Use subagents only when the user asks or independent work will materially
  shorten the task. Give each one a bounded scope; the parent owns decisions,
  and workers do not spawn more workers.

## Changes

- Check `git status --short` before editing and preserve unrelated work.
- Prefer the existing owner module and direct call path. Avoid copied
  entrypoints, monkey patches, generic wrapper layers, and duplicate defaults.
- Use `apply_patch` for authored file edits.
- Before deleting or overwriting material data, resolve the exact target and
  confirm that the request authorizes it. Never recursively target a home,
  repository, scratch root, or unresolved variable/glob.
- After deletion, report the exact removed scope and whether it is recoverable.
- Do not submit/cancel jobs, send messages, deploy, or mutate other external
  state unless the user asked for that action.
- Before cancel, submit, or retry, resolve the exact job and input identity from
  current state. Do not create a retry while an equivalent job is active.
- Authorization and readiness are separate. A request to perform an external
  action authorizes that action, but does not make incomplete validation
  sufficient or waive an unnamed production-path difference.
- Do not treat “checks are green,” prior approval, or an instruction to skip
  inspection as current readiness evidence for a consequential external action.
  If the required non-mutating verification is forbidden, block the action and
  state the missing evidence. Proceed with a gap only after the specific
  unexecuted difference is identified and explicitly accepted.

## Evidence And Validation

- Separate observed facts from inference. For live jobs, distinguish scheduler
  state from log/artifact completeness.
- State the exact claim each check supports. Evidence applies only to the code,
  data, control, and environment path it actually exercised. Never combine
  component-local, config-only, mocked, or manually adapted checks into proof
  of integrated behavior unless their composition is itself exercised.
- Derive the validation boundary backward from the exact result being claimed
  or the external action being considered. Include every input reader,
  transformation, branch, environment, and state transition whose failure
  would invalidate that claim; do not define the boundary by whichever check
  is already convenient.
- Before a consequential external action or fan-out, run the smallest
  non-mutating production-equivalent canary through that boundary. A material
  unexecuted difference blocks the action unless the user explicitly accepts
  that named difference. If only a mutating canary can close the gap, it
  requires its own authorization.
- A canary may stop before later expensive work only after the last material
  boundary needed for the proposed action has direct evidence.
- To recover a prior Codex source, resolve thread identity from
  `~/.codex/state_5.sqlite`, then verify exact anchors and first/terminal turns
  in its JSONL. Copied recap text alone is not identity evidence.
- Run the cheapest direct check that can catch a relevant defect. Add another
  check only for a distinct, consequential boundary.
- For a generated artifact, prefer loading it with its real reader over checking
  only that the file exists.
- Do not automatically add full test suites, smoke runs, independent reviews,
  manifests, hashes, schema layers, or provenance machinery.
- Stop when the requested behavior and the changed high-risk boundary have
  direct evidence. Report skipped checks only when their absence leaves a real
  residual risk.

## Prior Issue Notes

- Before changing or retrying a benchmark or pipeline, search
  `llm_wiki/catalog.py` with Codegraph for entries matching the current dataset
  and component.
- Apply a linked note only when its dataset, parser or staging owner, and input
  contract match the current work. Treat notes from another dataset or contract
  as hypotheses, then inspect the current caller and artifacts before reusing a
  fix.
- Modify `llm_wiki/` only when the user explicitly asks to record or update an
  issue.

## Response

- Lead with the result. Keep status updates and final answers concise.
- For changes, name the files changed, checks run, and any remaining material
  risk. Do not require ceremonial trace blocks.
