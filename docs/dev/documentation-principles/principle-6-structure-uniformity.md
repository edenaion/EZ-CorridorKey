# Principle 6: Documentation structure uniformity

Consistency in structure helps readers navigate confidently and maintainers scale without confusion.

## File Naming

Use lowercase with hyphens only.

**DO:**

- `layer-1-docker.md`
- `bump-api.md`
- `environment-variables-guide.md`

**DON'T:**

- `Layer1Docker.md`
- `BumpApi.md`
- `Environment_Variables.md`

## Folder Structure

Every folder with 2 or more docs needs an `index.md` as a navigation hub. Single-document folders should flatten to the parent directory.

`index.md` should:

- List all documents in the folder with one-line descriptions
- Be the entry point for readers discovering the folder
- Use consistent naming (`index.md`, not `README.md`)

## Document Structure

Every document must:

- Start with `# Title` matching the navigation label in `zensical.toml`
- Begin with a sentence establishing context and relationship to the system
- Use standard sections in this order:
  1. Purpose or Overview
  2. How It Works
  3. Steps or Configuration
  4. Key Concepts
  5. Examples (optional)
  6. Permissions or Requirements
  7. Next Steps or Related Documents

## Documentation Domains

**Thinking and Architecture** (`docs/thinking/`)

- Narratives, architectural decisions, core philosophies (e.g., ADT as Single Source of Truth).

**Data Formats** (`docs/formats/`)

- Specifications for inputs (DOCX anatomy), intermediate structures (Snapshot and ADT Schemas), and strict target outputs (JATS XML, LaTeX).

**Pipeline Stages** (`docs/pipeline/`)

- Algorithmic logic and execution details for distinct operational stages (Unpacker, Snapshot, ADT Builder, Emitters).

**Developer Guide & Reference** (`docs/dev/`)

- Tools, local setup, testing strategies, adding publisher templates, and package documentation.
