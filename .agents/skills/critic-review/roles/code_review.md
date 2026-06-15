# Code Review Checklist

Use this checklist for ordinary code, config, test, and script diffs that are
not primarily data-pipeline or harness changes. For data-pipeline code, use
`data_pipeline_review.md` as primary and this file as a cross-check.

## Checks

- Correctness bugs, behavioral regressions, race conditions, and bad error
  handling.
- Public API, config, CLI, schema, and default behavior compatibility.
- Tests or smoke checks are focused on the changed behavior and proportional to
  risk.
- Existing local utilities, patterns, and ownership boundaries are preserved.
- The implementation does not introduce unnecessary abstraction, dead code,
  abandoned paths, or broad rewrites.
- Organization-sensitive edits include a defensible organization-boundary,
  owner, naming, and abstraction
  decision. Flag as blocking when new or moved code lacks callsite/owner
  evidence, caller-specific behavior is hidden in a generic utility module
  without rationale, names obscure responsibility or consumer contract, or a
  public config/CLI/Hydra/sbatch surface expands without a repeated callsite,
  consumer isolation need, or explicit user request.
- New classes, dataclasses, wrappers, helper modules, or orchestration objects
  have a named reason: repeated call sites, a real boundary, a local pattern, or
  consumer-contract isolation. If they mostly forward arguments, mirror config,
  create a second source of truth, or replace a direct command path, flag them.
- New wrappers or orchestrators do not mostly forward arguments, duplicate
  config, create an alternate execution path, or expand schemas beyond what the
  consumer contract reads.
- Environment setup, install docs, and operational scripts agree on names,
  activation paths, and manual-vs-automated steps; machine-local assumptions are
  explicit without encoding private paths or credentials.
- Validation commands and skipped checks are reported honestly.

## Maintainability / Under-Structuring

Counterweight to the over-abstraction checks above: the same review must also catch code that
has too *little* structure, not only too much. These default to `Non-blocking` and must follow
the proportionality rule (smallest concrete fix, never a broad rewrite); escalate only when the
flagged code is the primary artifact under review or the fragility is demonstrated.

- Long parameter lists and stamp coupling: a call threading many positional/keyword arguments,
  or a large config object passed only to read two or three fields. Suggest grouping a stable
  argument cluster, but do not require a new dataclass for an argument set that is not repeated.
- Primitive-obsession config plumbing: deeply nested string-keyed access (`a["b"]["c"]["d"]`)
  threaded across function boundaries where a mistyped key fails deep or silently. Prefer
  dotted-path access (`OmegaConf.select(cfg, "a.b.c", default=...)` where the code path already
  uses OmegaConf) or a typed accessor. Flag
  as a demonstrated fragility (escalates) when a key is actually mistyped or duplicated.
- Reinvented idioms: hand-rolled `None`-ladders or re-implemented stdlib/OmegaConf behavior that
  a single existing call would replace.
- Readability: comments restate *what* the next line does instead of *why* (the constraint or
  intent); names obscure the responsibility or consumer contract.
- Change amplification: for a likely near-term change to this code (e.g. adding a config field or
  a new noise/data channel), name how many files must change in lockstep. A high count surfaces
  hidden duplication or a missing single source of truth without prescribing a specific
  abstraction.

## Operational / HPC / Sbatch UX

- Verify sbatch array size matches matrix dimensions and step/model/config
  counts.
- Check `#SBATCH` resources, job name, logs, output paths, environment
  activation, container/project root, and scratch paths against nearby working
  scripts.
- Prefer simple in-file matrices for small operational launchers when the user
  asks for a simple script.
- For sbatch or operational scripts, experiment-defining values should have one
  visible source-of-truth block; derived names, counts, list paths, labels, and
  logs should come from that block rather than being copied by hand.
- Confirm directly runnable UX: submit command, manual command, validation
  command, and expected output location are clear.
- Use proportional validation such as `bash -n`, dry-run config composition, or
  focused import/smoke checks; do not require broad tests for a narrow launcher
  edit.
- Treat optional CLI generalization, extra submit-time args, and broad refactors
  as non-blocking only when they are not the main implementation, the requested
  direct command/config path still works, and validation covers that path. If
  they replace or obscure the direct path, duplicate config, or introduce an
  unvalidated alternate execution path, report a blocking scope/correctness
  finding.
