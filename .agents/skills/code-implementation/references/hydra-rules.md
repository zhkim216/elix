# Hydra Rules

## Keep Config Resolution Stable

- Keep `@hydra.main(config_path=..., config_name=...)` valid after edits.
- Keep config group names synchronized with actual folder structure.
- Ask before changing default `config_name` values in entrypoints.

## Sync `configs/` And `configs_local/`

- Mirror shared keys when both trees provide the same logical config.
- If a key exists only in local debug configs, gate access with safe lookup such as `cfg.get(...)`.
- If introducing a required key, add it to all relevant variants or provide a backward-compatible default.

## Avoid Breaking Dotted Access

Do not remove or rename keys referenced as `cfg.a.b.c` without updating callsites, YAML files, and downstream assumptions. Ask the user first if intent is unclear.

## Preserve Reproducibility Defaults

Keep explicit seed, logging path, and deterministic settings stable unless the user asks for a change.
