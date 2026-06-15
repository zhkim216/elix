# Common Implementation Role

Use this role for every code-implementation task before selecting a primary role.

## Checks

- Start from the user-visible behavior and the existing consumer contract.
- For mask, tensor, batch-key, runtime-schema, conditioning, loss, or graph
  edits, lock the producer, consumer, default behavior, changed behavior,
  loss/graph implications, and proving assertion before editing.
- Before editing, inspect the relevant callsites, module ownership, and local
  helper APIs; summarize the chosen insertion point and any rejected abstraction.
- After any implementation-file edit, include a `RELATED_FILE_TRACE` in the
  final response with edited targets and at least one related non-target
  producer, writer, entrypoint, consumer, or callsite contract when available.
- For organization-sensitive edits, record a `CODE_ORGANIZATION_TRACE` that
  explains the owner, name, and abstraction boundary instead of only describing
  the code diff. Use the hook field names directly: `chosen_owner`,
  `helper_reuse_decision`, `naming_decision`, and
  `abstraction_decision`.
- Keep code readable: named helpers for repeated logic, direct control flow, no hidden global state, and no unnecessary framework layers.
- Structure balance cuts both ways. A typed object/dataclass, a dotted-path accessor
  (`OmegaConf.select`), or a small named struct is the right fix — not over-engineering — when it
  removes a long parameter list, stamp coupling, or deeply nested string-keyed config threaded
  across function boundaries. Justify it exactly as you justify removing a layer: it simplifies
  the current code path (see `roles/repo_code.md`). Do not reflexively avoid warranted structure.
- Prefer local patterns and existing APIs over custom parsers or new abstractions.
- Make validation proportional to risk: import/syntax checks for narrow edits, focused tests or smoke runs for behavior, and integrity checks for generated artifacts.
- Preserve unrelated work in dirty checkouts; stage or touch only files needed for the task.

## Anti-Patterns

- One large script body with interleaved parsing, transformation, I/O, and reporting.
- Copy-pasted logic that will drift across local and Sherlock paths.
- Generic classes, config systems, or staging layers created for one real call site.
- Generic `utils` modules or vague helper names chosen before inspecting the
  actual function mix and call graph.
- New helpers or modules added without first checking existing helper functions,
  nearby owners, and sibling module boundaries.
- Runtime artifacts carrying provenance columns or extra schema fields that the consumer does not read.
- Final responses after implementation edits that omit required trace markers or
  only describe the diff without naming the producer-consumer contract.
