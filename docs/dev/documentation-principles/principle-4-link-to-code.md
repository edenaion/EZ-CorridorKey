# Principle 4: Link to code properly

Never duplicate large code blocks. Link to the source instead. Code changes frequently; docs do not.

## Linking to Documentation Files

Use relative paths within the `docs/` folder.

**DO:**

- `[Documentation Principles](index.md)`
- `[Authoring Guide](../../dev/authoring-documentation.md)`

**DON'T:**

- Relative paths to code outside docs: `[config](../../pyproject.toml)`
- Broken anchor attempts to non-doc files

## Linking to Code and Non-Documentation Files

Use full GitHub permalinks. Static documentation sites cannot resolve relative paths outside `docs/`.

**DO:**

- `https://github.com/nikopueringer/CorridorKey/blob/main/pyproject.toml`
- `https://github.com/nikopueringer/CorridorKey/blob/main/zensical.toml`

**DON'T:**

- Relative paths: `[setup.sh](../../scripts/setup.sh)`
- Links without protocol or host: `blob/main/file.py`

## Referencing Git History and Decisions

When writing "Thinking" or "Journey" docs, link to specific commit SHAs.

**DO:**

- "In [commit a1b2c3d](https://github.com/owner/repo/commit/a1b2c3d), we increased replicas to 3 to handle peak traffic"
