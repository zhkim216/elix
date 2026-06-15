---
name: code-implementation-prompt-writer
description: Draft concise, decision-complete prompts to hand code, config, script, test, refactor, or validation tasks to $code-implementation in allatom-design. Use for prompt writing only; do not implement files.
---

# Code Implementation Prompt Writer

Use this skill when the user wants a prompt that can be given to
`$code-implementation`. The output is the handoff prompt, not the
implementation.

## Hard Rule

Do not edit code, configs, scripts, tests, generated artifacts, or harness files
while using this skill. If the prompt target, acceptance criteria, runtime
constraints, artifact paths, or validation expectations are unclear in a way
that changes correctness or cost, ask before drafting. Label low-impact
assumptions.

## Prompt Principles

Follow OpenAI prompt-guidance style for coding-agent handoffs:

- Put the outcome first: target behavior, success criteria, constraints, and
  available context before implementation ideas.
- Keep the prompt compact. Add details only when they prevent a concrete
  implementation mistake.
- Preserve exact user-provided literals: paths, commands, metric names,
  thresholds, sample counts, CCD lists, output names, branches, and runtimes.
- Use strong words such as `must` and `do not` only for real invariants, safety
  rules, accepted scope boundaries, or required output fields.
- Give a retrieval or inspection budget when useful: name the likely files,
  call paths, configs, or artifacts to inspect first, and say when enough
  evidence is enough.
- Require proportional validation: syntax/import checks for narrow edits,
  focused pytest or smoke checks for behavior, Hydra composition for configs,
  `bash -n` for shell/sbatch, and consumer-path checks for generated inputs.
- Do not duplicate the full `$code-implementation` skill. The prompt should
  supply task-specific intent and constraints; `$code-implementation` supplies
  the execution discipline.

## Workflow

1. Identify the implementation task category: package code, config/Hydra,
   script, test, refactor, operational wrapper, generated runtime input, or
   validation-only.
2. Separate discoverable repo facts from user preference. Inspect local files
   non-mutating only when the prompt needs concrete paths, callsites, schemas,
   or validation commands.
3. Capture the latest user source of truth. If a correction rejected an older
   assumption, include the corrected value and omit the stale one.
4. Draft one prompt that starts with `$code-implementation` and then gives
   `Goal`, `Context`, `Constraints`, `Done when`, `Implementation notes`,
   `Validation`, and `Assumptions`.
5. Keep implementation notes at behavior level unless a specific file, function,
   config key, schema field, or command is needed to prevent ambiguity.
6. Include a `CODE_ORGANIZATION_TRACE` request when the task may add or move a
   function/class/module, change an import owner, edit shared helper/config/io
   modules, change public-ish CLI/Hydra/sbatch names, or add a wrapper.
7. If a `$critic-review` pass is warranted, ask for it in the handoff only when
   the task changes a public API/config/schema, hook/harness behavior, metric
   semantics, generated-data contract, or broad cross-module behavior.

## Output Format

For ordinary cases, output only this prompt block:

```markdown
$code-implementation

Goal
[One or two sentences describing the user-visible behavior.]

Context
[Known repo facts, relevant paths, inspected call paths, or user-provided
background. Keep this short.]

Constraints
[Hard invariants, preserved defaults, out-of-scope work, dirty-checkout
warnings, and exact user literals.]

Done when
[Concrete acceptance criteria.]

Implementation notes
[Likely insertion points or approach. Include rejected wrappers or new
abstractions only when relevant.]

Validation
[Commands or check classes to run. Do not invent nonexistent lint/typecheck
commands.]

Assumptions
[Low-impact assumptions, or "None" if there are none.]
```

If important information is missing and cannot be discovered locally, ask the
smallest necessary question instead of outputting a prompt.

## Quality Gate

Before returning the prompt, check that it:

- can be executed by `$code-implementation` without inventing major behavior;
- names the concrete behavior and acceptance criteria;
- preserves user-provided literals exactly;
- avoids unnecessary wrappers, config copies, schema expansion, or framework
  layers;
- tells the implementer what to inspect first when repo facts matter;
- includes validation that matches the file types and risk;
- makes skipped checks reportable with reasons;
- does not ask for another plan unless the user explicitly wants plan-first
  behavior.

## Boundaries

- Do not implement the prompt.
- Do not add hooks or scripts for this workflow.
- Do not browse OpenAI docs on every use; use the embedded rules unless the user
  asks for the latest guidance or the prompt depends on changed OpenAI behavior.
- Do not save the prompt to the wiki unless the user explicitly asks.
