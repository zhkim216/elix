# Data Utility Role

Use this role for scripts that generate datasets, manifests, caches, converted inputs, or large batches of files.

## Implementation Shape

- Build a direct pipeline: discover inputs, validate/match records, transform records, write outputs, write a compact report.
- Split repeated logic into small pure helpers; keep orchestration in one clear `main`.
- Make outputs rerunnable with `--limit`, `--overwrite` or skip-existing behavior, and deterministic ordering.
- Use chunked or process-pool work for CPU-bound large batches; avoid one future per row.

## Validation

- Prove a smoke subset before full runs.
- Validate source counts, output counts, representative rows/files, duplicate handling, and skipped/error records.
- Keep provenance and diagnostics in a sibling manifest/report unless the runtime consumer needs them.
- Check for partial runs, stale outputs, truncation, and leftover temp files before handoff.
