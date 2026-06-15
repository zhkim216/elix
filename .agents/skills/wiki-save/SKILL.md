---
name: wiki-save
description: Save user-requested code notes, grounded answers, paper summaries, decisions, and reusable findings into the repo-local Codex wiki. Use only when the user explicitly asks to save, remember, archive, or add something to the wiki.
---

# Wiki Save

Use this skill only on explicit user request, such as "save this", "wiki에 남겨줘", or "remember this for later".

## Storage Layout

Use date-scoped folders based on the local command `date +%F`.

- Wiki pages: `.codex/wiki/pages/YYYY-MM-DD/<slug>.md`
- Raw papers: `.codex/raw/papers/YYYY-MM-DD/`
- Raw notes: `.codex/raw/notes/YYYY-MM-DD/`
- Raw results: `.codex/raw/results/YYYY-MM-DD/`
- Wiki index: `.codex/wiki/index.md`
- Save log: `.codex/wiki/log.md`

## Workflow

1. Ask the user before saving if scope, source trust, privacy sensitivity, title, category, or slug is unclear or suspicious.
2. Pick a short lowercase slug, using hyphens, for example `ligandmpnn-validation-notes`.
3. Save synthesized material as a Markdown page under the current date folder.
4. Prefer storing paths or links to raw materials. Copy raw files into `.codex/raw/<kind>/YYYY-MM-DD/` only when the user explicitly asks or confirms; do not rewrite papers or generated artifacts.
5. Update `.codex/wiki/index.md` with a one-line link to the page.
6. Append `.codex/wiki/log.md` with the date, slug, source paths, and reason for saving.

## Page Template

```markdown
# Title

## Summary
Short statement of what was saved and why it matters.

## Sources
- `path/or/url`

## Notes
- Key reusable facts, decisions, or commands.

## Follow-Ups
- Optional next actions.
```

## Guardrails

- Do not save secrets, private credentials, API keys, or machine-specific tokens.
- Do not save private dataset paths, unpublished sensitive results, or proprietary source text in tracked wiki pages unless the user explicitly confirms.
- Cite repo files, raw files, or external URLs for factual claims.
- Mark uncertain material as uncertain instead of presenting it as established.
- Do not invoke this skill from another skill unless the user explicitly requested persistence.
- Raw payloads under `.codex/raw/` are local by default and ignored by git; keep durable synthesized notes in `.codex/wiki/`.
