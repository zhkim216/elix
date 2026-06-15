---
name: code-implementation
description: Implement, fix, refactor, or validate code, Hydra configs, scripts, and tests in allatom-design. Use for code-writing tasks, not for pure Q&A, brainstorming, wiki saving, or figure-only work.
---

# Code Implementation

Use this skill when the user asks to change code, configs, scripts, tests, or repository behavior.

## Hard Rule

Follow the repo `User-Question Gate`: if requirements, acceptance criteria, behavior intent, API/config compatibility, data assumptions, validation feasibility, or runtime constraints are unclear or suspicious, ask the user before editing. Treat inferred schemas for artifacts consumed by an existing runtime/tool as suspicious until the consumer contract has been inspected. Low-impact assumptions must be labeled.

## Design Discipline

Keep implementations simple, modular, and direct. Avoid spaghetti control flow,
copy-pasted business logic, and clever abstractions that do not serve a real
repeated use. When a small utility is actually needed, it should have clear
inputs, a few named helper functions, explicit validation, and one obvious path
from source data to the consumer contract.

Prefer existing helper functions, local APIs, and nearby module patterns before
writing new code. If a new helper is needed, first place it in the closest
existing owner module that already carries the responsibility; create a new
`.py` file only after checking the surrounding folder structure and sibling
module boundaries.

Add an abstraction only when it removes meaningful duplication, isolates a real boundary, or matches an established local pattern. Keep defaults and labels close to the code path that uses them. Do not introduce configuration classes, generic staging layers, broad provenance frameworks, or reusable helper APIs for a single operational script unless the user explicitly asks or multiple independent workflows need the same non-entrypoint logic. This forbids *speculative* structure, not warranted structure: a typed object or dotted-path accessor whose concrete unmet requirement is "collapse an existing long signature or replace fragile nested string-keyed config access" is a simplification, and is exempt — match the under-structuring counterweight in `$critic-review` (`roles/common.md` "Structure balance", `roles/code_review.md` "Maintainability / Under-Structuring").

Treat concrete user-specified literals such as paths, CCD lists, sample indices,
batch size, walltime, metric radii, and output names as acceptance criteria.
Preserve them verbatim unless the user approves a change; when a label or count
must be derived, derive it mechanically from those literals and report the
derivation.

## Role Routing

Read `roles/common.md` for every implementation task, then read one primary role file:

- `roles/repo_code.md`: package code, tests, non-operational scripts, and refactors.
- `roles/data_utility.md`: generated datasets, manifests, batch conversion, caches, and large-corpus utilities.
- `roles/operational_wrapper.md`: local/Sherlock launchers, sbatches, smoke wrappers, command adapters, and scripts whose main job is to run or parameterize an existing entrypoint.

If the task spans roles, choose the role that owns the riskiest output and use the other as a cross-check. Role files are instructions, not evidence sources.

## Workflow

1. Restate the intended behavior and acceptance criteria.
2. Run `git status --short` before editing; ask before touching files with unrelated or suspicious existing changes.
3. Inspect relevant files with `rg`, focused reads, and the references below, then write a concise reconnaissance brief before editing. The brief must name the intended behavior, acceptance criteria, existing entrypoint or consumer contract, relevant modules/callsites inspected, local helpers or nearby patterns found, chosen insertion point, and any new helper/module/class/wrapper being introduced with its concrete reason. When changing masks, tensors, batch keys, runtime schemas, conditioning semantics, losses, or graph construction, first write a short contract lock: producer/writer, consumer/reader, default behavior, changed behavior, loss/graph implications, and the focused assertion or smoke that will prove the contract. When creating or modifying an artifact consumed by an existing entrypoint/tool, trace the consumer first: entrypoint, reader/helper, exact keys/columns/fields used, and tolerated optional fields. For Sherlock launchers, compare at least one nearby known-working sbatch/config before editing. Record the minimal runtime interface before generating the artifact.
   For organization-sensitive edits, include a `CODE_ORGANIZATION_TRACE` in the reconnaissance brief or final response. This applies when adding or moving a function/class, creating a Python module, changing an import owner, editing shared `utils`/`cfg`/`setup`/`io`/`runner`/`wrapper` modules, changing public-ish config/CLI/Hydra/sbatch names, or adding a wrapper/orchestrator. The trace must name the changed surfaces, responsibility, existing owners checked, options considered, chosen owner, helper reuse decision, naming decision, abstraction decision, and final decision.
4. Before writing custom parsers or format readers for repo-native data, inspect repo-local libraries and editable installs first. Prefer established local APIs and verify them with a small import/read smoke test on representative files.
5. Before adding a new wrapper, copied entrypoint, dataclass/config object,
   extra module, or orchestration layer, state the existing direct path and the
   unmet requirement. If the workflow can be expressed by the existing
   entrypoint plus CLI, Hydra, or sbatch values, edit that path instead. Note the
   converse unmet requirement is also valid: "the existing direct path itself is
   the problem (a long signature, or `a["b"]["c"]["d"]` threaded across layers)"
   justifies a typed object or accessor that simplifies that path.
6. Ask before proceeding if any design tradeoff or behavior change is ambiguous.
7. Implement the smallest coherent diff that follows existing local patterns. For operational launchers, this usually means a source-of-truth config plus a thin command wrapper, not a helper framework.
8. Preserve public APIs, Hydra keys, output schemas, and default metrics unless explicitly approved. Do not expand runtime input schemas with source/provenance fields unless the consumer reads them or the user requested them.
9. Before finalizing operational or script changes, inspect the diff shape:
   files touched, new files, new classes/dataclasses, copied entrypoints,
   duplicated source-of-truth, and wrapper depth. If any were added, report why
   they are necessary or simplify before validation. Self-review against the same
   canonical criteria the critic uses — `$critic-review`'s `roles/common.md` and
   `roles/code_review.md` (the checks `code_review.md` now points to), including
   the Maintainability / Under-Structuring block — so the diff is not minimized
   into a long signature or nested-dict plumbing the critic will then flag.
10. Validate touched files with `.agents/skills/code-implementation/scripts/run_targeted_checks.sh <file>...` and explicit pytest targets when risk warrants it. For shell/sbatch edits, run `bash -n`; for Hydra/config edits, run the closest config composition, dry-run, or `--cfg job` check when feasible; for Python entrypoints, run an import or module-mode smoke. For generated operational inputs, validate the consumer path or closest pure reader/helper on a tiny representative fixture when feasible.
11. Before the final response, run the implementation trace gate. If code,
   config, script, test, hook, skill, or harness files were edited, include a
   `RELATED_FILE_TRACE` with edited targets and the related non-target
   producer, writer, entrypoint, consumer, or callsite contract inspected. If
   the change is organization-sensitive, also include `CODE_ORGANIZATION_TRACE`.
   Then summarize changed files, behavior, validation commands, skipped checks,
   and residual risk.
12. For broad or generated-artifact work, follow the selected role file's smoke, resumability, and integrity gates before full execution.

## Repair Contract Guard

For missing-file, missing-helper, removed-symbol, or corrupted-code repairs,
reconstruct the local contract before editing. Trace direct callers and
entrypoints, search for required symbols, imports, tests, configs, docs,
exported symbol lists, comments, placeholders, TODOs, expected result formats,
and data-shape assumptions, and inspect at least one sibling helper or class
with the same API role when available. Confirm expected arguments, return
values, mutation behavior, defaults, ordering, side effects, exception behavior,
edge cases, and nearby helper patterns. Write the smallest implementation
supported by that evidence; do not invent optional modes, broad fallback
handling, extra parameters, logging, wrappers, compatibility layers, or generic
convenience semantics unless the observed call path requires them.

Treat validation as necessary but not sufficient. After validation passes,
reread the edited code and diff against retrieved local evidence, adjacent
functions, sibling patterns, caller expectations, and discovered call paths.
Confirm signatures, accepted input types, return shapes, mutation semantics,
defaults, imports, aliases, parsing behavior, ordering, error types and
messages, framework-specific formulas, data-preservation details, and
documentation style match observed usage. Remove or narrow unsupported
docstrings, wrappers, abstractions, broad type handling, compatibility guesses,
fallback branches, or extra edge-case behavior that is only speculative; if the
validator is broad or shallow, add one focused smoke check for the specific
helper behavior when dependencies allow.

## Operational Input Schema Guard

For existing runtime inputs such as `sampling_inputs.csv`, config fragments, JSON payloads, or name lists, do not infer the schema from the source dataset. Inspect the consumer and generate the smallest accepted artifact. Example: before writing `sampling_inputs.csv` for the sampling entrypoint `eval/sampling/run_elix.py`, check its readers — `prepare_sample_dict` and `resolve_query_pn_unit_iids` in `eval/utils/data_utils.py`, and `get_pdb_files` in `eval/utils/eval_setup_utils.py`; source provenance columns are not automatically part of the runtime contract.

## Tensor And Batch Contract Guard

For mask, tensor, batch-key, runtime-schema, conditioning, loss, or graph edits,
lock the data contract before implementation. Name the tensor or key being
changed, the function that writes it, the function that reads it, the default
behavior that must remain true, the selected or modified case, any loss or graph
side effect, and the exact assertion or smoke check that will prove the intended
contract. If the user's correction rejects an earlier assumption, restate the
latest contract before editing and do not revive the rejected default.

## Refactor And Naming Guard

For refactors, module moves, helper extraction, and naming changes, inspect the
actual function mix and call graph before choosing a new boundary or name. Do
not use broad buckets such as `utils`, `data`, `misc`, or opaque numeric suffixes
unless that is already the local pattern and the inspected responsibility really
matches. Prefer names that expose the unit of iteration, consumer contract, or
domain responsibility.

Before adding a new helper module, first identify the closest existing module
that owns the behavior and explain why editing that module is insufficient. If a
new helper only serves one caller, keep it local to the caller unless the local
codebase already has a matching shared-helper pattern. Before merging look-alike
helpers, verify the semantic convention against the source-of-truth constant or
the actual consumer before unifying them.

Use this trace shape for organization-sensitive edits:

```text
CODE_ORGANIZATION_TRACE:
- target_files:
- changed_surfaces:
- responsibility:
- existing_owners_checked:   (behavior-based repo-wide scan across folders, not just nearest module)
- options_considered:
- chosen_owner:
- helper_reuse_decision:     (reuse/consolidate into the canonical owner when a behavior-equivalent helper exists)
- naming_decision:
- abstraction_decision:
- decision:
```

Use this trace shape after implementation-file edits:

```text
RELATED_FILE_TRACE:
- target_files:
- changed_values:
- entrypoint_or_consumer:
- producer_or_writer_contract:
- existing_convention:
- decision:
```

## Focused Commit And Push Guard

Before commit or push work, inspect `git status --short`, `git remote -v`, and
the target branch relationship. Stage only the files requested or directly
required by the task. In a dirty checkout, leave unrelated changes untouched. If
the user asks to fold changes into the previous pushed commit, amend or squash
and push with `--force-with-lease`; otherwise create a focused normal commit.
After a history rewrite, give pull/reset/rebase commands using the actual remote
and branch names you verified, not a remembered alias. Report staged files,
commit action, push action, and validation.

## Critic Triggers

Use `$critic-review` for cross-module behavior changes, public API/config/schema changes, metric definition changes, failed validation, hook/harness edits, or any suspicious requirement.

## Repo References

- `references/codebase-map.md`: primary modules and routing rules.
- `references/hydra-rules.md`: config compatibility rules.
- `references/validation-matrix.md`: validation selection.
