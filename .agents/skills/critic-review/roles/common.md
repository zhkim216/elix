# Common Critic Checklist

Use this checklist for every `critic-review` task before applying any
role-specific checklist.

## Universal Scope

- Review only the requested artifacts unless the user asks for a broader repo
  audit.
- Do not edit files.
- Do not overrule explicit user constraints.
- Do not require broad rewrites when a narrow fix satisfies the goal.
- If the review target, expected standard, source of truth, or acceptance
  criteria is unclear or suspicious, ask before issuing a confident review.
- Label low-impact assumptions.

## Universal Checks

- Correctness bugs and behavioral regressions.
- Missing validation, focused tests, reproducibility evidence, or acceptance
  proof.
- Unsupported claims, weak citations, or evidence that does not support the
  stated conclusion.
- Scope creep, hidden behavior changes, compatibility risks, and public contract
  drift.
- Minimality: for code, plan, harness, or operational reviews, name the existing
  direct path, list added wrappers, state, schemas, or source-of-truth copies,
  and classify each as necessary, optional, or reject.
- Structure balance: minimality cuts both ways. Also name under-structuring
  fragility — long signatures or stamp coupling, untyped nested string-keyed
  config flow that fails deep on a typo, and duplicated defaults — and classify
  each as necessary-fix, optional, or ignore. These are not Blocking by default,
  but surface them as named findings; do not let them dissolve into "naming
  polish".
- Whether the work followed the repo `User-Question Gate`.
- Whether the work added unnecessary gates that block deterministic safe
  progress.
- Whether explicit user constraints, requested scope, and source-of-truth
  boundaries were preserved.

## Severity Contract

- `Blocking`: correctness bug, violated user constraint, wrong artifact/path,
  unsafe or destructive action, missing validation for behavior that cannot
  otherwise be trusted, or a primary wrapper/orchestrator/harness path that
  lacks a concrete boundary over an existing direct command/config path,
  duplicates config or schema state, obscures the source of truth, or prevents
  proportional validation of the requested behavior.
- `Blocking` for shape/scope: technically working code or plans that violate an
  explicit user constraint for simple, direct, or non-bloated work; revive a
  rejected abstraction; mostly forward arguments through a new layer; hide
  source-of-truth values; or make validation less direct.
- `Blocking` for demonstrated under-structuring: a mistyped or duplicated nested
  config key that fails silently, or a default duplicated across surfaces that
  has already drifted out of sync, is a correctness bug — not naming polish — and
  blocks. (Long signatures or untyped nested-dict flow that is merely fragile but
  not yet broken stays Non-blocking; see the next bullet.)
- `Non-blocking`: cleanup, ergonomics, optional abstraction, optional wrapper
  cleanup when the requested behavior and direct path are already clear and
  validated, broader test coverage, naming polish, future-proofing, or docs not
  needed for the requested task. Named under-structuring fragility (long
  signatures, stamp coupling, untyped nested string-keyed config flow, duplicated
  defaults that have not yet drifted) is Non-blocking but must still be reported
  as a named finding — it is not "naming polish" and must not be silently dropped.
- Non-blocking simplification findings should propose the smallest concrete
  simplification, not a broader redesign.
- Required fixes should include only blocking findings. Non-blocking findings
  may be reported but must not prevent handoff.

## Anti-Recursion Rule

A critic review must not request another critic review by default. If fixes are
needed, name the concrete fix and validation command, then stop. A second review
is only for explicit user request, broad/high-risk rewrite, failed validation, or
unresolved blocking uncertainty.

## Output Contract

- Lead with findings, ordered by severity.
- Ground findings in file paths, lines, artifact paths, commands, or evidence
  references when available.
- Then list open questions, required fixes, and residual risks.
- If no issues are found, say so clearly and still mention test gaps or residual
  risk.
