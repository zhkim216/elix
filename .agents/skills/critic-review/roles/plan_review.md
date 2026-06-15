# Plan Review Checklist

Use this checklist for implementation plans, design docs, scope decisions, and
acceptance criteria. For harness or data-pipeline plans, use this as a cross
check with the relevant domain role.

## Checks

- The goal, audience, success criteria, in-scope work, and out-of-scope work are
  clear.
- The plan is decision-complete enough for another engineer or agent to execute
  without inventing major behavior.
- Assumptions and defaults are explicit, especially where the user did not
  choose a tradeoff.
- Interfaces, schemas, commands, files, or outputs are specified only as much as
  needed for safe implementation.
- Plans that add a copied entrypoint, mostly-forwarding wrapper, orchestrator,
  harness, or alternate execution path must name the direct existing
  command/config path inspected and the concrete unmet requirement that makes
  the new layer necessary. If the reason is naming, convenience, or passing
  through values, recommend the direct path and report scope bloat.
- The approach is not over-specified with unnecessary policy, validation, or
  migration detail.
- For multi-iteration work, check whether the plan revives a previously rejected
  abstraction, stale default, old path, or superseded source-of-truth choice.
- Edge cases, failure modes, and test/acceptance criteria match the risk of the
  requested change.
- The plan does not defer a critical decision to the implementer.
