# Codebase Map

## Primary Paths

- `allatom_design/train_seq_denoiser.py`: sequence denoiser training entrypoint.
- `allatom_design/model/`: model architecture, denoisers, and losses.
- `allatom_design/data/`: `datasets/`, `preprocessing/`, `transform/`, and `utils/`.
- `allatom_design/eval/`: evaluation entrypoints and sampling pipelines.
- `allatom_design/eval/utils/`: shared evaluation helpers and metrics.
- `allatom_design/utils/`: cross-cutting helpers, including `checkpoint_utils.py`.
- `allatom_design/configs/`: baseline Hydra configs used broadly.
- `allatom_design/configs_local/`: local/debug configs and environment-specific overrides.

## Routing Rules

| Change Intent | Start Here | Also Inspect |
|---|---|---|
| Training behavior or checkpointing | `train_*.py`, `model/*`, `utils/checkpoint_utils.py` | matching config groups and resume logic |
| Sequence design sampling/eval | `eval/sampling/*`, `eval/utils/*` | `eval/utils/*`, AF3 helper paths |
| Eval/sampling input artifacts | consuming `eval/sampling/*` entrypoint | file discovery helpers, DataFrame filters, and exact reader functions in `eval/utils/*` |
| Data parsing or writing | `data/datasets/*`, `data/transform/*`, `data/utils/*` | callsites in train/eval entrypoints |
| Metric logic | `eval/utils/*` (metric helpers) | metric consumers and logging keys |
| Hydra config changes | `configs/*` or `configs_local/*` | matching counterpart tree and defaults in entrypoints |

## Common Couplings

- Entry scripts and config names: `@hydra.main(config_path=..., config_name=...)` must match real files.
- Eval pipelines and output schema: CSV/metric key changes can break downstream plotting scripts.
- Runtime input artifacts are contracts with their consumers; source-dataset provenance belongs in a manifest/report unless the consumer reads it.
- Resume/checkpoint flow: training scripts often reconstruct run directories from checkpoint path.
- AF3 workflow: sampling and AF3 phases may be intentionally separated.
