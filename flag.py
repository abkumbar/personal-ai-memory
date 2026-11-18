#!/usr/bin/env python3
"""Add keyword-based sensitivity flags to memory notes.

Guarantees:
- Flags are only ever added, never removed.
- `keyword_flags` records every flag this script has added. If you delete one of
  those from `flags`, it is not added back.
- Only the `flags` and `keyword_flags` properties change. The body is untouched.
- Text in the Source section (the conversation title) is not scanned.

Standard library only.
"""

import argparse
from collections import Counter
import json
from pathlib import Path
import re
import sys

import ingest

RULES_PATH = Path(__file__).resolve().parent / "flag_rules.json"


def load_rules(path=RULES_PATH):
    rules = json.loads(path.read_text(encoding="utf-8"))
    return {flag: [re.compile(pattern, re.IGNORECASE) for pattern in patterns]
            for flag, patterns in rules.items() if not flag.startswith("_")}


def scanned_text(lines, body):
    title = next((line[7:] for line in lines if line.startswith("title: ")), "")
    return title + "\n" + body.split("\n## Source", 1)[0]


def match_flags(text, rules):
    return {flag for flag, patterns in rules.items() if any(p.search(text) for p in patterns)}


def read_list(lines, key):
    """Return (values, start, end) for a flow or block YAML list; start is None if absent."""
    for index, line in enumerate(lines):
        match = re.match(rf"{re.escape(key)}:(.*)$", line)
        if not match:
            continue
        rest = match.group(1).strip()
        if rest:
            items = rest.strip("[]")
            values = [v.strip().strip("\"'") for v in items.split(",") if v.strip()]
            return values, index, index + 1
        end = index + 1
        values = []
        while end < len(lines) and re.match(r"\s*- ", lines[end]):
            values.append(lines[end].split("- ", 1)[1].strip().strip("\"'"))
            end += 1
        return values, index, end
    return [], None, None


def write_list(lines, key, values):
    _, start, end = read_list(lines, key)
    line = f"{key}: {json.dumps(values)}"
    if start is None:
        return lines + [line]
    return lines[:start] + [line] + lines[end:]


def flag_note(text, rules):
    """Return (new text, flags added). New text equals the input when nothing changes."""
    lines, body = ingest.split_note(text)
    if lines is None:
        return text, set()
    flags, _, _ = read_list(lines, "flags")
    already_added, start, _ = read_list(lines, "keyword_flags")
    matched = match_flags(scanned_text(lines, body), rules)
    added = matched - set(flags) - set(already_added)
    if not added:
        return text, set()
    recorded = sorted(set(already_added) | added)
    lines = write_list(lines, "flags", flags + sorted(added))
    lines = write_list(lines, "keyword_flags", recorded)
    return ingest.join_note(lines, body), added


def run(folder, rules, dry_run=False):
    stats = Counter()
    for path in sorted(folder.glob("*.md")):
        text = path.read_text(encoding="utf-8")
        lines, _ = ingest.split_note(text)
        if not lines or "type: memory" not in lines:
            continue
        stats["memories"] += 1
        before, _, _ = read_list(lines, "flags")
        new_text, added = flag_note(text, rules)
        for flag in added:
            stats[f"added: {flag}"] += 1
        if added:
            stats["memories newly flagged" if not before else "memories given extra flags"] += 1
            if "access: approved" in lines:
                stats["APPROVED memories that gained a flag"] += 1
        if new_text != text and not dry_run:
            ingest.write_atomic(path, new_text)
    prefix = "[dry run] " if dry_run else ""
    for key, count in sorted(stats.items()):
        print(f"{prefix}{key}: {count}")
    return stats


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--vault", type=Path, default=Path.home() / "personal_chats")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    run(args.vault / "03-Memories", load_rules(), args.dry_run)
    return 0


if __name__ == "__main__":
    sys.exit(main())
