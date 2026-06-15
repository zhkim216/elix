# Data Pipeline Review Checklist

Use this checklist for data generation, caches, manifests, provenance, large
corpus processing, and generated datasets.

## Checks

- Source-of-truth drift, fallback precedence, and cache invalidation inputs are
  explicit and correct.
- Existing manifests, sampling tables, ledgers, or cache indexes are reused as
  the source of truth when their row semantics match the task. New derived
  manifests are justified by a semantic change, not by convenience, and preserve
  the selector criteria used to create them.
- Generated artifact semantics are clear: row meaning, field meaning, units,
  source layer, and whether rows are raw, intermediate, or derived.
- Under-structured artifacts (the data-side counterweight to schema sprawl): a
  single overloaded delimited string field where typed columns are expected, an
  untyped field with no unit or source-layer, or a default duplicated across
  cache/manifest layers that can drift. Flag as a named fragility; Blocking only
  when the drift or silent failure is demonstrated.
- Provenance is preserved through manifests, input paths, digests, row keys, or
  source references as appropriate.
- Output schema, field counts, representative rows, duplicate handling, and
  integrity checks are validated.
- Large-corpus or networked workflows use safe concurrency, rate limiting,
  resumability, and smoke/sample checks before full runs.
- Final output-integrity checks are sufficient to catch partial runs, stale
  caches, truncation, or missing rows.
