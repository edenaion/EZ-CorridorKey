"""Guard: no raw cv2 file I/O outside the unicode-safe facade.

cv2.imread / cv2.imwrite / cv2.VideoCapture / cv2.VideoWriter use a narrow
(ANSI codepage) file API on Windows and silently fail on non-ASCII paths. All
image/video disk access must route through backend.frame_io (imread_unicode,
imwrite_unicode, open_video). This test parses the AST of every module under
backend/ and ui/ and fails if any raw cv2 file call survives, so the fix can
never silently regress when new code is added.

AST parsing (not text grep) means cv2 calls inside comments, docstrings, or
string literals are correctly ignored.
"""
import ast
import os

import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCAN_DIRS = ["backend", "ui"]

# cv2 functions that take a filesystem path and break on non-ASCII paths.
BANNED_ATTRS = {"imread", "imwrite", "VideoCapture", "VideoWriter"}

# The facade itself is the single allowed home for raw cv2 file calls.
ALLOWED_FILES = {os.path.normpath("backend/frame_io.py")}


def _python_files():
    for scan in SCAN_DIRS:
        base = os.path.join(REPO_ROOT, scan)
        for root, dirs, files in os.walk(base):
            dirs[:] = [d for d in dirs if d not in ("__pycache__", "_BACKUPS")]
            for fname in files:
                if fname.endswith(".py"):
                    yield os.path.join(root, fname)


def _cv2_aliases(tree: ast.AST) -> set[str]:
    """Collect every local name bound to the cv2 module in a parsed file."""
    aliases: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == "cv2":
                    aliases.add(alias.asname or "cv2")
    return aliases


def _violations_in(path: str) -> list[tuple[int, str]]:
    with open(path, "r", encoding="utf-8") as f:
        tree = ast.parse(f.read(), filename=path)
    aliases = _cv2_aliases(tree)
    if not aliases:
        return []
    hits: list[tuple[int, str]] = []
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr in BANNED_ATTRS
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id in aliases
        ):
            hits.append((node.lineno, f"{node.func.value.id}.{node.func.attr}"))
    return hits


def test_no_raw_cv2_file_io():
    offenders: list[str] = []
    for path in _python_files():
        rel = os.path.normpath(os.path.relpath(path, REPO_ROOT))
        if rel in ALLOWED_FILES:
            continue
        for lineno, call in _violations_in(path):
            offenders.append(f"{rel}:{lineno}  {call}(...)")
    assert not offenders, (
        "Raw cv2 file I/O found outside backend/frame_io. Route these through "
        "imread_unicode / imwrite_unicode / open_video so non-ASCII paths work:\n"
        + "\n".join(offenders)
    )
