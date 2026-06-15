# Repo Code Role

Use this role for package code, tests, ordinary scripts, and refactors that are not primarily data generation or operational launchers.

## Implementation Shape

- Keep the diff focused on the behavior requested.
- Use existing module boundaries and helper APIs before adding new ones.
- Before introducing or recommending a new helper during refactor/dedup/review,
  scan the repo by behavior across folders (not just the nearest module) for
  existing equivalents and consolidate into the canonical owner.
- If a new helper is needed, put it in the nearest existing owner when that
  owner already carries the responsibility; create a new module only after
  checking the folder structure and sibling module responsibilities.
- Add abstractions only when they simplify the current code and have a clear owner.
- For refactors and helper extraction, name the actual responsibility and unit
  of iteration; avoid broad utility buckets unless the inspected call graph
  supports that boundary.
- When adding or moving code ownership, compare the local private-helper option,
  the nearest existing owner module, and a new-module option before choosing.
  Names should expose responsibility, iteration unit, or consumer contract.
- Preserve public APIs, config keys, output schemas, and metric definitions unless the user approved the change.

## Validation

- Run the smallest meaningful pytest target or import smoke.
- Add focused tests for reusable behavior or regression-prone changes.
- For config-backed behavior, verify both true and false paths when users can toggle the setting.
