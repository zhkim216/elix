---
name: brainstorm-planner
description: Help with brainstorming, planning, design tradeoffs, research strategy, and decision-complete implementation plans before code changes. Use for plan/brainstorm prompts and do not edit files.
---

# Brainstorm Planner

Use this skill when the user asks to brainstorm, plan, compare approaches, design a workflow, or think through strategy. This is the default for plan-oriented prompts before implementation; if the user asks to plan then implement, route to `$code-implementation` with an explicit planning phase.

## Hard Rule

Follow the repo `User-Question Gate`: if goals, constraints, success criteria, risk tolerance, audience, available data, or user preferences are unclear or suspicious, ask the user before converging on a plan. Low-impact assumptions must be labeled.

## Operational Fast Path

For repo/script/config operational tasks with a clear existing entrypoint, produce a one-pass plan with labeled assumptions instead of entering clarification loops. Ask only when the answer changes correctness, safety, cost, or target artifact selection. Defer optional enhancements into a short "Later" note; do not make them prerequisites.

When the user corrects values or rejects a design across iterations, first
collapse the latest state into a short summary: source of truth, derived values,
stale assumptions now rejected, and next command or artifact. Do not revive
earlier defaults, wrapper choices, paths, or config ownership unless the user
asks to revisit them.

Start operational plans by identifying the existing consumer contract, current entrypoint/config/script path, and final runtime command or sbatch invocation. If that path can express the requested workflow with config, Hydra, CLI, or sbatch values, make direct execution the primary plan and include only the necessary command/config edits. Treat wrappers or orchestrators as "Later" or "Rejected unless needed" unless you can name the unmet requirement: a missing direct parameter, a required consumer-contract change, scheduler/environment setup that cannot stay readable as direct invocation, multiple independent workflows that need the same adapter, or an explicit user request.

For operational scripts and configs, name the intended source of truth before
proposing derived files or helper layers. If values can live in one existing
sbatch/config and derived names can be computed there, prefer that over mirrored
config state.

When a wrapper is justified, keep its boundary thin: environment setup, path/config selection, scheduler matrices, command echoing, or bounded smoke defaults.

Structure balance cuts both ways. When the direct path would require a long positional signature, deep string-keyed config plumbing that fails late on a typo, a None-ladder, or duplicated defaults, name the warranted structure (a typed object, a single named boundary) as a *necessary* part of the plan rather than reflexively deferring it. Reject only unjustified wrappers and copies; do not reject structure that removes a concrete maintainability smell. This mirrors the under-structuring counterweight `$critic-review` will apply downstream.

If the user requested plan then implementation, hand off to `$code-implementation` after the first sufficient plan. For operational plans, end with a concrete implementation contract: existing entrypoint, source of truth, acceptance criteria (Done-when), preserved user literals/constraints, files expected to change, structural decisions (rejected wrapper/copy options *and* any warranted structure with the maintainability smell it removes), and validation command — aligning to the `Goal / Context / Constraints / Done-when` skeleton in the root `AGENTS.md`. This contract is the decision record, not a formatted prompt; if the user wants a paste-ready implementation prompt, route to `$code-implementation-prompt-writer`. If a new file is proposed, state why the existing entrypoint cannot cover it. Recommend `$critic-review` once for high-risk or explicitly requested review, but do not recommend repeated critic cycles for ordinary operational edits unless the first critic finds blocking issues.

## Workflow

1. Separate goals, known constraints, unknowns, and non-goals.
2. Inspect repo context when the plan depends on local implementation details.
3. Offer concrete options with tradeoffs when multiple designs are plausible.
4. Ask targeted questions for high-impact unknowns.
5. Produce a decision-complete plan only after enough information is available.
6. Recommend `$critic-review` for plans involving hooks, public APIs, experiment conclusions, broad refactors, or explicit user request.
7. End with decision points, a recommended path, unresolved questions, and one handoff state: `Ready to implement`, `Blocked pending user answer`, or `Plan-only complete`.

For long, multi-step, or handoff-heavy plans, use the root `PLANS.md` structure
so the final plan records context inspected, decisions, implementation shape,
test plan, and review plan.

## Boundaries

- Do not edit files or run mutating commands.
- Do not present uncertain assumptions as decisions.
- Do not invoke `$wiki-save` unless the user explicitly asks to persist the plan.
