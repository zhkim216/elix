# Validation Matrix

Use this matrix after implementing a code change.

## Baseline Check

Run:

```bash
.agents/skills/code-implementation/scripts/run_targeted_checks.sh <changed-file>...
```

If `python` is not available on `PATH`, or a documented environment command is
missing, do not stop validation there. Inspect available workspace interpreters
(for this repo, also check `/home/yjhk/model-dev/envs/uv/*/bin/python`) and rerun
the same command with that environment's `bin` directory prepended to `PATH`.
Report both the failed interpreter attempt and the interpreter that actually ran
the checks.

## Additional Checks By Area

| Touched Area | Additional Check |
|---|---|
| `allatom_design/train_*` | Compile check plus relevant config sanity check. |
| `allatom_design/eval/glide/*` | Use `--with-tests` to run matching glide tests if available. |
| `allatom_design/data/*` around residue/cache logic | Use `--with-tests` for matching data/cache tests if available. |
| Hydra YAML configs | YAML syntax check plus compile check for Python readers. |
| Generated operational input artifacts | Validate the target consumer path or closest pure loader/reader on a tiny representative fixture; check required keys/columns only, tolerated optional fields, and representative rows. If the full runtime needs Sherlock, checkpoints, or large data, document the skipped runtime check and the local substitute validation. |
| Shell scripts | `bash -n` syntax check through the targeted helper. |
| Environment setup scripts or install docs | Verify documented activation/install commands in the target shell or container when feasible; run focused import/CLI smoke checks from a clean cwd; for patched external tools, confirm the expected runtime flag or behavior from inside the target environment. |
| Shared utility modules | Prefer one direct call-path sanity check if environment permits. |

When `--with-tests` prints that no mapping matched, choose an explicit pytest target or explain why no focused test exists.

## Report Template

- Changed files:
- Commands executed:
- Passed checks:
- Skipped or blocked checks and reason:
- Residual risk:
