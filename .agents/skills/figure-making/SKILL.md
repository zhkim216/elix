---
name: figure-making
description: Create, revise, or validate figures, plots, charts, paper figures, and visual analysis outputs for allatom-design with reproducible data handling.
---

# Figure Making

Use this skill when the user asks for a figure, plot, chart, visualization, paper figure, or regenerated visual analysis output.

## Hard Rule

Follow the repo `User-Question Gate`: if input data, metric definitions, filtering, denominators, labels, visual target, output path, or interpretation is unclear or suspicious, ask the user before producing or modifying the figure. Low-impact assumptions must be labeled.

## Workflow

1. Identify source data paths and expected output files.
2. Inspect data with code, not the rendered image, for numeric claims.
3. Print or record the data funnel: row counts, filters, denominators, missing values, and grouped sample sizes.
4. Write or reuse a reproducible local analysis script near the analysis/debug context. Route shared package or production script behavior changes through `$code-implementation`.
5. Save figures to the requested location or a clearly named debug/output directory.
6. Verify that output files exist and the script reruns from source data.
7. Report generated paths, source data, commands, and any visual or statistical caveats.

## Quality Checks

- Axis labels, units, legends, titles, and captions must match the data.
- Avoid silently dropping rows; explain filters and NaN handling.
- For paper-facing figures, use `$critic-review` before final handoff.

## Do Not

- Read exact numerical conclusions from a rendered plot.
- Overwrite existing output files without checking first. Ask before overwriting unless the target is clearly disposable debug output.
- Save figure notes to wiki unless the user explicitly requests `$wiki-save`.
