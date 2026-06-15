---
name: critic-review
description: Review plans, code changes, grounded answers, figures, wiki entries, skills, hooks, and harness changes for correctness, evidence, scope, and risk. Use when explicitly requested or when another skill flags high-risk uncertainty.
---

# Critic Review

Use this skill for explicit review requests and for high-risk work from other skills. Review only the requested artifacts unless the user asks for a broader repo audit.

## Hard Rule

Follow the repo `User-Question Gate`: if the review target, expected standard, source of truth, or acceptance criteria is unclear or suspicious, ask the user before issuing a confident review. Low-impact assumptions must be labeled.

## Role Checklists

Use `roles/common.md` for every review, then choose one primary role checklist.
For mixed targets, add the named cross-check role checklist. Role files are
review instructions, not evidence sources. The parent skill owns the trigger
boundary, source-of-truth clarity, role routing, requested-artifact scope,
output format, no-edit boundary, and final response synthesis.

Available role checklists:

```text
roles/common.md
roles/plan_review.md
roles/code_review.md
roles/data_pipeline_review.md
roles/harness_skill_review.md
roles/grounded_answer_review.md
roles/final_artifact_review.md
```

Primary role selection:

- Use `harness_skill_review.md` for skills, hooks, AGENTS files, Codex harnesses,
  and multi-agent workflows.
- Use `data_pipeline_review.md` for data generation, caches, provenance,
  manifests, large-corpus processing, and generated datasets.
- Use `code_review.md` for ordinary code, config, test, and script diffs that
  are not primarily data-pipeline or harness changes.
- Use `plan_review.md` for implementation plans, design docs, scope decisions,
  and acceptance criteria.
- Use `grounded_answer_review.md` for factual answers, local evidence claims,
  citations, and uncertainty handling.
- Use `final_artifact_review.md` for saved reports, figures, wiki entries, and
  final user-facing deliverables.

Mixed-target routing examples:

- Harness or skill implementation plan: primary `harness_skill_review.md` plus
  cross-check `plan_review.md`.
- Operational workflow or sbatch plan/code: primary `code_review.md`; add
  `plan_review.md` while execution shape is still undecided, and add
  `data_pipeline_review.md` only when generated manifest or data semantics are
  part of the reviewed artifact.
- Data-pipeline plan: primary `data_pipeline_review.md` plus cross-check
  `plan_review.md`.
- Data-pipeline code diff: primary `data_pipeline_review.md` plus cross-check
  `code_review.md`.
- Code diff producing a final report or artifact: primary `code_review.md` plus
  cross-check `final_artifact_review.md`.
- Grounded answer saved as a wiki/report: primary `grounded_answer_review.md`
  plus cross-check `final_artifact_review.md`.
- Ordinary plan, code, answer, or artifact with no domain overlap:
  `common.md` plus one primary role only.

## Output Format

Lead with findings, ordered by severity. Include file paths or evidence references where possible. Then list open questions, required fixes, and residual risks. If no issues are found, say so and note remaining test gaps.

## Boundaries

- Do not edit files.
- Do not overrule explicit user constraints.
- Do not require broad rewrites when a narrow fix satisfies the goal.
