"""Sign a release manifest for the in-app updater.

Reads dist/SHA256SUMS.txt (produced by the build scripts), wraps it into
a JSON manifest, and signs it with the offline Ed25519 private key.

Usage:
    python scripts/sign_release.py --key path/to/ezck_signing_private.pem

Output:
    dist/manifest.json       (version + file hashes)
    dist/manifest.json.sig   (Ed25519 signature, raw 64 bytes)
"""
from __future__ import annotations

import argparse
import json
import sys
from getpass import getpass
from pathlib import Path

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey


def main() -> None:
    parser = argparse.ArgumentParser(description="Sign a release manifest")
    parser.add_argument(
        "--key", required=True, type=Path,
        help="Path to ezck_signing_private.pem",
    )
    parser.add_argument(
        "--dist", default=Path("dist"), type=Path,
        help="Directory containing SHA256SUMS.txt and release artifacts",
    )
    args = parser.parse_args()

    sums_path = args.dist / "SHA256SUMS.txt"
    if not sums_path.exists():
        print(f"ERROR: {sums_path} not found. Run the build first.")
        sys.exit(1)

    # Parse SHA256SUMS.txt into file list
    files = []
    for line in sums_path.read_text().strip().splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split(None, 1)
        if len(parts) != 2:
            print(f"WARNING: skipping malformed line: {line}")
            continue
        sha256, name = parts
        # Strip leading path separators (e.g. *filename on some sha256sum outputs)
        name = name.lstrip("*").strip()
        files.append({"name": name, "sha256": sha256.lower()})

    if not files:
        print("ERROR: no entries in SHA256SUMS.txt")
        sys.exit(1)

    # Read version from pyproject.toml
    pyproject = Path("pyproject.toml")
    version = "unknown"
    if pyproject.exists():
        import tomllib
        with open(pyproject, "rb") as f:
            version = tomllib.load(f)["project"]["version"]

    manifest = {
        "version": version,
        "files": files,
    }

    # Canonical JSON encoding (sorted keys, no extra whitespace)
    manifest_bytes = json.dumps(
        manifest, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")

    # Load private key. Skip the passphrase prompt entirely when the PEM
    # is unencrypted, because cryptography rejects a non-None password
    # on an unencrypted key (and Windows terminals can return stray
    # whitespace from a bare Enter, triggering that rejection).
    key_bytes = args.key.read_bytes()
    head = key_bytes[:200]
    encrypted = (
        b"BEGIN ENCRYPTED PRIVATE KEY" in head
        or b"Proc-Type: 4,ENCRYPTED" in key_bytes
    )
    if encrypted:
        typed = getpass("Signing key passphrase: ").strip()
        passphrase = typed.encode() if typed else None
    else:
        passphrase = None
    try:
        priv_key = serialization.load_pem_private_key(
            key_bytes, password=passphrase
        )
    except Exception as e:
        print(f"ERROR: could not load private key: {e}")
        sys.exit(1)

    if not isinstance(priv_key, Ed25519PrivateKey):
        print("ERROR: key is not Ed25519")
        sys.exit(1)

    signature = priv_key.sign(manifest_bytes)

    # Write outputs
    manifest_path = args.dist / "manifest.json"
    sig_path = args.dist / "manifest.json.sig"
    manifest_path.write_bytes(manifest_bytes)
    sig_path.write_bytes(signature)

    print(f"Manifest: {manifest_path}")
    print(f"  Version: {version}")
    print(f"  Files:   {len(files)}")
    for f in files:
        print(f"    {f['sha256'][:16]}...  {f['name']}")
    print(f"Signature: {sig_path} ({len(signature)} bytes)")
    print("Done. Upload manifest.json + manifest.json.sig with the release.")


if __name__ == "__main__":
    main()
