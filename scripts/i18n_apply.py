"""Apply a JSON translation mapping to a .ts catalogue.

Used by the translation pipeline: an agent produces a JSON object
mapping English source strings to translated strings, this script
fills every unfinished entry deterministically and validates that
printf-style placeholders survive translation intact.

Usage:
    python scripts/i18n_apply.py ui/translations/corridorkey_de.ts de.json
"""
from __future__ import annotations

import json
import re
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

_PLACEHOLDER_RE = re.compile(r"%(?:\d+|[sd])")


def placeholders(text: str) -> list[str]:
    return sorted(_PLACEHOLDER_RE.findall(text))


def main() -> int:
    if len(sys.argv) != 3:
        print(__doc__)
        return 2
    ts_path = Path(sys.argv[1])
    mapping = json.loads(Path(sys.argv[2]).read_text(encoding="utf-8"))

    tree = ET.parse(ts_path)
    root = tree.getroot()

    filled = 0
    missing: list[str] = []
    bad_placeholders: list[str] = []

    for ctx in root.findall("context"):
        for msg in ctx.findall("message"):
            tr = msg.find("translation")
            if tr is None:
                continue
            unfinished = tr.get("type") == "unfinished"
            empty = not (tr.text or "").strip()
            if not (unfinished or empty):
                continue
            source = msg.findtext("source") or ""
            if source not in mapping:
                missing.append(source)
                continue
            translated = mapping[source]
            if placeholders(source) != placeholders(translated):
                bad_placeholders.append(source)
                continue
            tr.text = translated
            if "type" in tr.attrib:
                del tr.attrib["type"]
            filled += 1

    if bad_placeholders:
        print(f"ERROR: placeholder mismatch in {len(bad_placeholders)} entries:")
        for s in bad_placeholders[:10]:
            print(f"  {s!r}")
        return 1
    if missing:
        print(f"ERROR: {len(missing)} unfinished entries missing from mapping:")
        for s in missing[:10]:
            print(f"  {s!r}")
        return 1

    ET.indent(tree, space="    ")
    header = '<?xml version="1.0" encoding="utf-8"?>\n<!DOCTYPE TS>\n'
    body = ET.tostring(root, encoding="unicode")
    ts_path.write_text(header + body + "\n", encoding="utf-8")
    print(f"OK: filled {filled} entries in {ts_path.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
