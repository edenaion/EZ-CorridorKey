# Contributing to EZ-CorridorKey

Thanks for your interest in improving EZ-CorridorKey! Whether you're a VFX artist, a pipeline TD, or a developer, contributions of all kinds are welcome — bug reports, feature ideas, documentation fixes, and code.

## Legal Agreement

EZ-CorridorKey is a GUI built on top of [Niko Pueringer's CorridorKey](https://github.com/nikopueringer/CorridorKey). By contributing to this project, you agree that your contributions will be licensed under the same terms as the upstream project's **[CorridorKey Licence](https://github.com/nikopueringer/CorridorKey/blob/main/LICENSE)**.

## Getting Started

### Prerequisites

- Python 3.11
- A virtual environment (`python -m venv .venv`)
- GPU with CUDA support (recommended), or Apple Silicon for MLX backend

### Dev Setup

```bash
git clone https://github.com/edenaion/EZ-CorridorKey.git
cd EZ-CorridorKey
python -m venv .venv
.venv/Scripts/activate      # Windows
source .venv/bin/activate    # macOS / Linux
pip install -r requirements.txt
```

### Running the App

```bash
python main.py --gui         # launch the desktop GUI
python main.py --cli         # original CLI wizard (upstream compatible)
```

### Running Tests

```bash
pytest                       # run all tests
pytest -v                    # verbose
pytest -m "not gpu"          # skip GPU-dependent tests
```

Most tests run in seconds and don't require a GPU or model weights.

### Linting and Formatting

```bash
ruff check                   # check for lint errors
ruff format --check          # check formatting
ruff format                  # auto-format
```

## Making Changes

### Pull Requests

1. Fork the repo and create a branch from `main`
2. Make your changes
3. Run tests and lint checks
4. Open a pull request against `main`

In your PR description, focus on **why** you made the change, not just what changed. If you're fixing a bug, describe the symptoms. If you're adding a feature, explain the use case.

### What Makes a Good Contribution

- **Bug fixes** — especially platform-specific issues (Windows, macOS, Linux), EXR/linear workflows, or color space handling
- **Tests** — more coverage is always welcome, particularly for the inference pipeline and clip management
- **Documentation** — better explanations, usage examples, or clarifying comments
- **Performance** — reducing VRAM usage, speeding up frame processing, or optimizing I/O
- **UI/UX** — improvements to the PySide6 GUI, accessibility, or workflow ergonomics

### Code Style

- [Ruff](https://docs.astral.sh/ruff/) for linting and formatting
- Line length: 120 characters
- Third-party model code in `gvm_core/`, `VideoMaMaInferenceModule/`, and `CorridorKeyModule/` is excluded from lint enforcement — those are kept close to their upstream research repos

### Model Weights

Model checkpoints (CorridorKey, GVM, VideoMaMa, MatAnyone2) are **not** in the git repo. Most tests don't need them. If you're working on inference code, follow the download instructions in the [README](README.md).

## Reporting Bugs

Open a [GitHub issue](https://github.com/edenaion/EZ-CorridorKey/issues) with:

- OS and GPU info
- Steps to reproduce
- Expected vs actual behavior
- Relevant log output (check `logs/` directory)

## Security Vulnerabilities

Please **do not** open public issues for security vulnerabilities. See [SECURITY.md](SECURITY.md) for responsible disclosure instructions.

## Questions?

Join the [Discord](https://discord.gg/TyxNjcWeF3) — it's the fastest way to get help or discuss ideas before opening a PR.
