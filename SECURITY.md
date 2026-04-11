# Security Policy

## Reporting a Vulnerability

If you discover a security vulnerability in EZ-CorridorKey, please report it responsibly through one of these channels:

1. **GitHub Private Security Advisory** (preferred): Go to the [Security tab](https://github.com/edenaion/EZ-CorridorKey/security/advisories/new) and create a private advisory.
2. **Email**: Send details to **EZ-CorridorKey@proton.me**

**Please do not** open public GitHub issues for security vulnerabilities.

## What to Include

- Description of the vulnerability
- Steps to reproduce
- Affected versions
- Potential impact

## Response Timeline

- **Acknowledgment**: Within 72 hours
- **Initial assessment**: Within 1 week
- **Fix or mitigation**: Depends on severity, but we aim for 30 days for critical issues

## Scope

The following are in scope:

- The EZ-CorridorKey Python application and GUI
- Model loading and inference pipeline
- Docker container configuration
- Build and packaging scripts (PyInstaller, NSIS)
- File I/O and subprocess handling

The following are **out of scope**:

- Third-party model weights hosted externally
- Upstream dependencies (report those to their maintainers)
- Issues requiring physical access to the machine
- Social engineering

## Security Practices

- All `torch.load()` calls use `weights_only=True` to prevent pickle deserialization attacks
- Subprocess calls use list-based arguments (no `shell=True`)
- No network-facing services in the desktop application
- Docker ports are bound to localhost only
