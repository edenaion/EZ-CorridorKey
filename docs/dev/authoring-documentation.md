# Authoring Documentation

Write effective documentation using Markdown and Zensical features. Follow [Documentation Principles](documentation-principles/index.md) for consistency.

## Current Docs Layout

The repository currently maintains the docs pages under `docs/` and `docs/dev/`.

- `docs/index.md` - Documentation home
- `docs/dev/authoring-documentation.md` - Authoring guidance
- `docs/dev/developer-setup.md` - Setup guidance
- `docs/dev/resources.md` - Tooling and references
- `docs/dev/documentation-principles/` - Standards and rules

## Markdown Essentials

**File structure:**

- One H1 per file (page title)
- H2 for sections, H3 for subsections (don't go deeper)
- Fenced code blocks with language identifiers

**Links & References:**

| Type | Format | Use When |
|------|--------|----------|
| Internal docs | `Link Text -> path/to/file.md` | Linking to other docs |
| Source code | Full GitHub link to the target file | Referencing code in passing |
| External | `[Link Text](https://url.com)` | Third-party resources |

For complete Markdown syntax, see [Zensical Reference](https://zensical.org/docs/authoring/markdown/).

## Admonitions (Callouts)

Use `???+` for collapsible blocks (expanded by default):

```markdown
???+ note "Optional Title"
    Block content here.
```

**Core types**: `note`, `tip`, `info`, `warning`, `danger`, `question`, `success`

**When to use:**

- Use for prerequisites, warnings, best practices, and status labels
- Avoid for regular content, minor points, and replacing structure

Example:

```markdown
???+ warning "Destructive Action"
    This will delete data permanently.
```

For advanced admonitions, see [Zensical Guide](https://zensical.org/docs/authoring/admonitions/).

## Tables

Use for structured data. Keep simple (3-5 columns ideal):

```markdown
| Column A | Column B | Column C |
|----------|----------|----------|
| Data 1   | Data 2   | Data 3   |
```

## Style Guidelines

| Guideline | Good | Avoid |
|-----------|---------|---------|
| **Voice** | Direct, active: "Run this command" | Passive: "Command should be run" |
| **Clarity** | Define acronyms once | Use undefined abbreviations |
| **Code examples** | Show context: `cd docs` then run command | Isolated commands without setup |
| **Consistency** | Use same terms throughout | Mix terminology |

## Before Committing

```bash
mise run docs
```

Verify:

- Broken links
- Formatting correct
- Code blocks render
- Admonitions display
- Tables aligned
- Follows documentation principles
- Matches current code (no fiction)

## See Also

- [Documentation Principles](documentation-principles/index.md) - Core standards
- [Zensical Markdown](https://zensical.org/docs/authoring/markdown/) - Full syntax reference
- [Zensical Admonitions](https://zensical.org/docs/authoring/admonitions/) - Advanced callouts
