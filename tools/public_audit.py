#!/usr/bin/env python3
"""Reject common private/large artifacts before a public release.

This is a lightweight guardrail, not a replacement for human review or a
secrets scanner. It intentionally ignores Python bytecode under __pycache__
because those files are excluded by .gitignore and may be created by tests.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path


FORBIDDEN_SUFFIXES = (".pyc", ".pyo", ".b64", ".gguf", ".safetensors", ".bin", ".zip", ".tar", ".tar.gz")
FORBIDDEN_NAMES = {"known_hosts", "dmit_ed25519", "id_rsa", "id_ed25519"}
PRIVATE_KEY = re.compile(r"-----BEGIN [^-]*PRIVATE KEY-----")
OPENSSH_KEY = re.compile(r"(?m)^ssh-(?:rsa|ed25519|ecdsa)\s+[A-Za-z0-9+/=]+(?:\s|$)")


def iter_files(root: Path):
    for path in root.rglob("*"):
        if not path.is_file() or "__pycache__" in path.parts or ".git" in path.parts:
            continue
        yield path


def audit(root: Path) -> list[str]:
    findings: list[str] = []
    for path in iter_files(root):
        relative = path.relative_to(root).as_posix()
        lower = relative.lower()
        if any(lower.endswith(suffix) for suffix in FORBIDDEN_SUFFIXES):
            findings.append(f"forbidden artifact: {relative}")
            continue
        if path.name in FORBIDDEN_NAMES or path.suffix.lower() in {".pem", ".key"}:
            findings.append(f"credential-like filename: {relative}")
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        if PRIVATE_KEY.search(text):
            findings.append(f"private key marker: {relative}")
        if OPENSSH_KEY.search(text):
            findings.append(f"embedded OpenSSH key: {relative}")
    return findings


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", nargs="?", type=Path, default=Path("."))
    args = parser.parse_args(argv)
    root = args.root.resolve()
    if not root.is_dir():
        parser.error(f"not a directory: {root}")
    findings = audit(root)
    if findings:
        print("Public audit failed:", file=sys.stderr)
        print("\n".join(f"- {finding}" for finding in findings), file=sys.stderr)
        return 1
    print(f"Public audit passed: {root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
