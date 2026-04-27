"""Generate an Ed25519 keypair for release manifest signing.

Run ONCE. Store the private key offline (never commit it).
The public key gets hardcoded into backend/update_verify.py.

Usage:
    python scripts/generate_signing_key.py

Output:
    ezck_signing_private.pem   (keep offline, passphrase-protected)
    ezck_signing_public.pem    (paste into backend/update_verify.py)
"""
from __future__ import annotations

from getpass import getpass
from pathlib import Path

from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from cryptography.hazmat.primitives import serialization


def main() -> None:
    priv = Ed25519PrivateKey.generate()
    pub = priv.public_key()

    passphrase = getpass("Passphrase for private key: ").encode()
    confirm = getpass("Confirm passphrase: ").encode()
    if passphrase != confirm:
        print("ERROR: passphrases do not match")
        return

    priv_path = Path("ezck_signing_private.pem")
    pub_path = Path("ezck_signing_public.pem")

    priv_path.write_bytes(
        priv.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.PKCS8,
            encryption_algorithm=serialization.BestAvailableEncryption(passphrase),
        )
    )
    pub_path.write_bytes(
        pub.public_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PublicFormat.SubjectPublicKeyInfo,
        )
    )

    print(f"Private key: {priv_path.resolve()}")
    print(f"Public key:  {pub_path.resolve()}")
    print()
    print("Paste this into backend/update_verify.py -> SIGNING_PUBLIC_KEY_PEM:")
    print(pub_path.read_text())


if __name__ == "__main__":
    main()
