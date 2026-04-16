"""Verify signed release manifests for the in-app updater.

The updater downloads manifest.json + manifest.json.sig from a GitHub
release, then calls verify_manifest() to confirm the signature matches
the baked-in public key. If valid, verify_file() checks each downloaded
artifact's SHA-256 against the manifest before applying the update.

If either check fails, the updater aborts without touching the install.
"""
from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path

from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey
from cryptography.hazmat.primitives import serialization
from cryptography.exceptions import InvalidSignature

logger = logging.getLogger(__name__)

# Paste the contents of ezck_signing_public.pem here after running
# scripts/generate_signing_key.py. This key ships with every installed
# copy of the app and is used to verify that updates came from edenaion.
#
SIGNING_PUBLIC_KEY_PEM = b"""-----BEGIN PUBLIC KEY-----
MCowBQYDK2VwAyEAeSFrBonqXDbDFPbrA/kpSe2OABoYttySeHnNt04aM+g=
-----END PUBLIC KEY-----"""


class UpdateVerificationError(Exception):
    """Raised when manifest signature or file hash verification fails."""


def verify_manifest(manifest_bytes: bytes, signature: bytes) -> dict:
    """Verify the Ed25519 signature on a manifest and return parsed JSON.

    Raises UpdateVerificationError if the signature is invalid or the
    manifest cannot be parsed.
    """
    try:
        pub_key = serialization.load_pem_public_key(SIGNING_PUBLIC_KEY_PEM)
    except Exception as e:
        raise UpdateVerificationError(f"Cannot load signing public key: {e}")

    if not isinstance(pub_key, Ed25519PublicKey):
        raise UpdateVerificationError("Signing key is not Ed25519")

    try:
        pub_key.verify(signature, manifest_bytes)
    except InvalidSignature:
        raise UpdateVerificationError(
            "Manifest signature is INVALID. The update may have been tampered with."
        )
    except Exception as e:
        raise UpdateVerificationError(f"Signature verification error: {e}")

    try:
        manifest = json.loads(manifest_bytes)
    except json.JSONDecodeError as e:
        raise UpdateVerificationError(f"Manifest JSON parse error: {e}")

    logger.info("Manifest signature verified OK (version %s)", manifest.get("version"))
    return manifest


def verify_file(path: Path, expected_sha256: str) -> None:
    """Verify a downloaded file's SHA-256 matches the manifest entry.

    Raises UpdateVerificationError on mismatch.
    """
    actual = hashlib.sha256(path.read_bytes()).hexdigest().lower()
    expected = expected_sha256.lower()
    if actual != expected:
        raise UpdateVerificationError(
            f"SHA-256 mismatch for {path.name}: "
            f"expected {expected[:16]}..., got {actual[:16]}..."
        )
    logger.info("SHA-256 verified OK: %s", path.name)


def get_expected_hash(manifest: dict, filename: str) -> str | None:
    """Look up a filename's expected SHA-256 in the manifest."""
    for entry in manifest.get("files", []):
        if entry.get("name") == filename:
            return entry.get("sha256")
    return None


def is_signing_key_configured() -> bool:
    """Check whether the placeholder key has been replaced."""
    return b"REPLACE_ME_WITH_REAL_KEY" not in SIGNING_PUBLIC_KEY_PEM
