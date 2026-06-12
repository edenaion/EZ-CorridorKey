"""Sync translation catalogues with the source code.

pyside6-lupdate only extracts literal self.tr() and
QCoreApplication.translate() calls. The main-window mixins use the
_tr() helper (ui/main_window_mixins/__init__.py), which lupdate cannot
see, so every mixin string silently dropped out of the catalogues in
the first 2.1.0 extraction. This script closes that gap:

1. Runs the documented pyside6-lupdate pass on every .ts file
2. AST-scans ui/ for _tr("...") literals (context: MainWindow)
3. Injects any missing sources into each .ts as unfinished entries
4. Reports per-language counts of untranslated strings
5. --release compiles every .ts to .qm with pyside6-lrelease

Usage:
    python scripts/i18n_sync.py            # lupdate + inject + report
    python scripts/i18n_sync.py --report   # report only, no writes
    python scripts/i18n_sync.py --release  # also compile .qm files
"""
from __future__ import annotations

import argparse
import ast
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
UI_DIR = ROOT / "ui"
TS_DIR = UI_DIR / "translations"
TR_CONTEXT = "MainWindow"  # _tr() resolves to this context at runtime


def find_tr_strings() -> set[str]:
    """AST-scan ui/ for _tr("literal") call arguments."""
    sources: set[str] = set()
    for py in UI_DIR.rglob("*.py"):
        if "__pycache__" in py.parts:
            continue
        try:
            tree = ast.parse(py.read_text(encoding="utf-8"))
        except SyntaxError as exc:
            print(f"WARN: cannot parse {py}: {exc}", file=sys.stderr)
            continue
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            name = getattr(func, "id", None) or getattr(func, "attr", None)
            if name != "_tr" or not node.args:
                continue
            arg = node.args[0]
            if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
                sources.add(arg.value)
    return sources


def run_lupdate(ts_file: Path) -> None:
    cmd = [
        "pyside6-lupdate", "-extensions", "py", "-recursive",
        str(UI_DIR), "-ts", str(ts_file),
    ]
    result = subprocess.run(
        cmd, capture_output=True, text=True,
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
    )
    if result.returncode != 0:
        raise RuntimeError(f"lupdate failed for {ts_file.name}:\n{result.stderr}")


def inject_tr_strings(ts_file: Path, sources: set[str]) -> int:
    """Add missing _tr() sources to the MainWindow context. Returns count added.

    Also resurrects entries lupdate marked vanished/obsolete: lupdate cannot
    see _tr() calls in the source, so every sync pass it declares those
    entries dead, and lrelease then drops them from the .qm. Stripping the
    type attribute keeps the translations live.
    """
    tree = ET.parse(ts_file)
    root = tree.getroot()

    context = None
    for ctx in root.findall("context"):
        if ctx.findtext("name") == TR_CONTEXT:
            context = ctx
            break
    if context is None:
        context = ET.SubElement(root, "context")
        ET.SubElement(context, "name").text = TR_CONTEXT

    resurrected = 0
    existing = set()
    for msg in context.findall("message"):
        src = msg.findtext("source")
        existing.add(src)
        if src in sources:
            tr = msg.find("translation")
            if tr is not None and tr.get("type") in ("vanished", "obsolete"):
                del tr.attrib["type"]
                resurrected += 1

    missing = sorted(sources - existing)
    for src in missing:
        msg = ET.SubElement(context, "message")
        ET.SubElement(msg, "source").text = src
        tr = ET.SubElement(msg, "translation")
        tr.set("type", "unfinished")

    if missing or resurrected:
        ET.indent(tree, space="    ")
        lang = root.get("language", "")
        header = (
            '<?xml version="1.0" encoding="utf-8"?>\n'
            "<!DOCTYPE TS>\n"
        )
        body = ET.tostring(root, encoding="unicode")
        ts_file.write_text(header + body + "\n", encoding="utf-8")
        # Sanity: must still parse
        ET.parse(ts_file)
        assert root.get("language", "") == lang
    return len(missing)


def report(ts_file: Path) -> tuple[int, int]:
    """Return (total messages, untranslated messages)."""
    root = ET.parse(ts_file).getroot()
    total = 0
    untranslated = 0
    for ctx in root.findall("context"):
        for msg in ctx.findall("message"):
            total += 1
            tr = msg.find("translation")
            if tr is None:
                untranslated += 1
                continue
            unfinished = tr.get("type") == "unfinished"
            empty = not (tr.text or "").strip()
            if unfinished or empty:
                untranslated += 1
    return total, untranslated


def run_lrelease(ts_file: Path) -> None:
    result = subprocess.run(
        ["pyside6-lrelease", str(ts_file)],
        capture_output=True, text=True,
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
    )
    if result.returncode != 0:
        raise RuntimeError(f"lrelease failed for {ts_file.name}:\n{result.stderr}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", action="store_true", help="report only, no writes")
    parser.add_argument("--release", action="store_true", help="compile .qm after sync")
    parser.add_argument("--skip-lupdate", action="store_true",
                        help="only inject _tr strings (skip the lupdate pass)")
    args = parser.parse_args()

    ts_files = sorted(TS_DIR.glob("corridorkey_*.ts"))
    if not ts_files:
        print(f"No .ts files in {TS_DIR}", file=sys.stderr)
        return 1

    tr_sources = find_tr_strings()
    print(f"_tr() strings in source: {len(tr_sources)}")

    for ts in ts_files:
        if not args.report:
            if not args.skip_lupdate:
                run_lupdate(ts)
            added = inject_tr_strings(ts, tr_sources)
        else:
            added = 0
        total, untranslated = report(ts)
        line = f"{ts.name}: {total} messages, {untranslated} untranslated"
        if added:
            line += f" (+{added} injected)"
        print(line)
        if args.release and not args.report:
            run_lrelease(ts)

    if args.release and not args.report:
        print("Compiled all catalogues to .qm")
    return 0


if __name__ == "__main__":
    sys.exit(main())
