# Harness And Skill Review Checklist

Use this checklist for skills, hooks, AGENTS files, Codex harnesses, and
multi-agent workflows.

## Checks

- The harness boundary is clear: trigger surface, non-trigger cases, ownership,
  and what remains human judgment.
- Proposed harnesses, wrappers, hooks, or multi-agent workflows first identify
  the direct existing CLI/config/script/checklist path inspected, including
  whether a small prepare/check utility would suffice. Flag as blocking when the
  new layer mostly forwards args, duplicates config or schema state, creates an
  alternate source of truth, or lacks a reusable boundary; keep it non-blocking
  only when the requested behavior is already satisfied and the layer is
  optional ergonomics.
- Parent skill and role files do not conflict. Role files are instructions, not
  evidence sources.
- Output path ownership, write scopes, and communication boundaries are explicit
  for every role.
- Multi-agent workflows preserve parent-owned integration and do not rely on
  peer-to-peer agent communication.
- Deterministic validation gates, provenance gates, and coverage gates are not
  weakened by the refactor.
- Hooks have deterministic predicates, finite retry budgets, fail-open behavior,
  and actionable continuation messages.
- Hooks that enforce organization or naming discipline check deterministic
  evidence markers only. They must not try to judge subjective design quality;
  that review belongs in `critic-review`.
- Hook or harness changes do not introduce false positives, deadlocks,
  fail-closed behavior, or bypass abuse.
- Harness docs stay compact and route detailed role behavior to role files when
  the workflow has multiple agent roles.
- Skill or harness refinements separate durable general rules from repo-local
  conventions and one-off project fixes, and place each rule in the narrowest
  file that owns it. Parent skills should stay as routing/workflow surfaces
  rather than accumulating role prompts, schemas, path manifests, or long
  command recipes.
