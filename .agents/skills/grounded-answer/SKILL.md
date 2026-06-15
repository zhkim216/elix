---
name: grounded-answer
description: Answer factual questions about allatom-design, local files, raw papers, data, biology, model behavior, or external facts with inspected evidence and explicit uncertainty. Use for Q&A, not code edits.
---

# Grounded Answer

Use this skill for factual answers when the user is asking, explaining, comparing, or verifying rather than asking for code edits.

## Hard Rule

Follow the repo `User-Question Gate`: if the question, source authority, data provenance, paper identity, biological claim, currentness, conflicting sources, or confidence is unclear or suspicious, ask the user before answering as fact. Low-impact assumptions must be labeled.

## Evidence Sources

- Repo code and configs: cite file paths and line numbers when practical.
- Local raw materials: inspect `.codex/raw/papers/`, `.codex/raw/notes/`, and `.codex/raw/results/`.
- Command outputs: run focused non-mutating commands for local facts.
- Web or external sources: browse when the user asks for latest/current facts, direct source attribution, or anything likely to have changed.

Treat `.codex/raw/notes/` and `.codex/raw/results/` as local artifacts, not authoritative sources, unless provenance is clear.

For read-only answer tasks, avoid evidence commands that write bytecode, caches,
or other artifacts. Prefer source inspection; if a Python smoke is necessary,
run it with bytecode/cache writing disabled, such as `PYTHONDONTWRITEBYTECODE=1`
or `python -B`, and disclose any skipped execution when mutation-free validation
is not feasible.

## Answer Rules

- Separate sourced facts from inference.
- Use citations such as `path/to/file.py:123`, `.codex/raw/papers/YYYY-MM-DD/name.pdf page N`, or source URLs.
- Do not invent paper claims, PDB biology, ligand roles, metric definitions, or experiment results.
- Mark weak claims as `UNCERTAIN`, `INFERRED`, or `SOURCE NEEDED`.
- Use `$critic-review` for high-impact scientific, biological, medical, legal, financial, or publication-facing claims.

## Do Not

- Edit files unless the user switches to an implementation request.
- Save to wiki unless the user explicitly asks for `$wiki-save`.
- Treat summaries, filenames, or rendered plots as sufficient evidence for numerical claims.
