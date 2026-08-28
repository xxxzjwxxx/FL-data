from __future__ import annotations

import re
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SELF = Path(__file__).resolve()
TEXT_SUFFIXES = {
    ".csv",
    ".json",
    ".md",
    ".py",
    ".toml",
    ".txt",
    ".yaml",
    ".yml",
}
PRIVATE_SUFFIXES = {".xls", ".xlsx", ".pt", ".pth", ".pkl", ".pickle", ".joblib"}
PRIVATE_DIRS = {"outputs", "results", "predictions"}
PATTERNS = {
    "remote workspace path": re.compile(r"/vepfs", re.IGNORECASE),
    "Windows user path": re.compile(r"[A-Za-z]:[\\/]Users[\\/]", re.IGNORECASE),
    "Linux home path": re.compile(r"/(?:home/[^/]+|root)/"),
    "IPv4 address": re.compile(
        r"(?<![\d.])(?:25[0-5]|2[0-4]\d|1?\d?\d)"
        r"(?:\.(?:25[0-5]|2[0-4]\d|1?\d?\d)){3}(?![\d.])"
    ),
    "institution identifier": re.compile(
        "\u8d35\u5dde\u533b\u79d1\u5927\u5b66\u9644\u5c5e\u533b\u9662"
    ),
}


def main() -> int:
    findings: list[str] = []
    scanned = 0
    for path in sorted(ROOT.rglob("*")):
        if not path.is_file() or ".git" in path.parts:
            continue
        relative = path.relative_to(ROOT)
        if path.suffix.lower() in PRIVATE_SUFFIXES:
            findings.append(f"private artifact type: {relative}")
        if any(part in PRIVATE_DIRS for part in relative.parts):
            findings.append(f"generated/private directory content: {relative}")
        if path == SELF or path.suffix.lower() not in TEXT_SUFFIXES:
            continue
        scanned += 1
        text = path.read_text(encoding="utf-8", errors="replace")
        for label, pattern in PATTERNS.items():
            if pattern.search(text):
                findings.append(f"{label}: {relative}")

    if findings:
        print("Release audit failed:")
        for finding in findings:
            print(f"- {finding}")
        return 1
    print(f"Release audit passed: {scanned} text files scanned.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
