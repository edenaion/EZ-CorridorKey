# Developer Setup

This page documents setup paths that match the repository as it exists today.

## Prerequisites

- [Git](https://git-scm.com/)
- [Mise](https://mise.jdx.dev/)
- Optional: [uv](https://docs.astral.sh/uv/) for dependency management

## Setup

1. Fork the repository.
2. Create a branch for your work.
3. Run `mise install`.
4. Run `mise run sync` to set up `uv` and install all dependencies.
5. Activate the virtual environment.

    === "Linux/MacOS"

        ``` sh
        source ./.venv/bin/activate
        ```

    === "Windows"

        ``` sh
        .venv/Scripts/Activate.ps1
        ```

## Pre-commit Hooks

Install hooks after dependencies are set up:

```bash
prek install
prek install --hook-type commit-msg
```

This enables automatic checks before each commit for:

- File format validation (JSON, YAML, TOML)
- Code formatting and linting (Python with Ruff)
- Commit message format (Conventional Commits)
- Dependency synchronization (UV lockfiles)

???+ tip "Warm up hooks"
    Run `prek run --all-files` immediately after installing the hooks. That preloads each environment and prevents surprises during your next commit.

## Related

- [Authoring Documentation](authoring-documentation.md)
- [Resources](resources.md)
- [Documentation Principles](documentation-principles/index.md)
