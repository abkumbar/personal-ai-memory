#!/usr/bin/env python3
"""Per-provider sync status: what is captured, how fresh it is, and where the gaps are.

  python3 status.py           print the report and write 05-Reports/Sync Status.md

Standard library only. Reads the vault, the archived exports and pipeline state.
"""

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

import extract
import ingest
import policy

PROVIDERS = ("chatgpt", "claude", "grok")
STALE_DAYS = 30
QUIET_GAP_DAYS = 45
UNIMPORTED = {
    "claude": "Claude exports also contain saved memories and projects (not imported).",
    "grok": "Grok exports also contain projects and tasks (not imported). Conversations from the "
            "Grok Bot desktop/iPhone app may not be included in the grok.com export.",
    "chatgpt": "ChatGPT exports also contain saved memories and custom instructions (not imported).",
}
CAPTURE = {
    "chatgpt": "Settings > Data controls > Export data (emailed link)",
    "claude": "Settings > Privacy > Export data (emailed link)",
    "grok": "grok.com > Settings > Data controls > Export (or accounts.x.ai)",
}


def days_between(a, b):
    return (b - a).days if a and b else None


def provider_status(vault, now=None):
    now = now or datetime.now(timezone.utc)
    extracted = json.loads((ingest.STATE_DIR / "extraction.json").read_text()) \
        if (ingest.STATE_DIR / "extraction.json").exists() else {}
    exports = dict(extract.latest_exports(vault))
    memories = Counter()
    for _, text in policy.memory_notes(vault):
        lines, _ = ingest.split_note(text)
        source = policy.value(lines, "source")
        memories[(source, "total")] += 1
        memories[(source, policy.value(lines, "access"))] += 1
        if policy.value(lines, "publish_status") == "published":
            memories[(source, "published")] += 1
    rows = []
    for source in PROVIDERS:
        row = {"source": source, "warnings": []}
        path = exports.get(source)
        if path is None:
            row["warnings"].append("No export imported yet.")
            rows.append(row)
            continue
        conversations = ingest.read_export(source, path)
        created = sorted(c["created"] for c in conversations if c["created"])
        newest = max((c["updated"] or c["created"] for c in conversations if c["updated"] or c["created"]), default=None)
        newest_conv = max(conversations, key=lambda c: c["updated"] or c["created"] or ingest.EPOCH, default=None)
        archived = datetime.strptime(path.name[:8], "%Y%m%d").replace(tzinfo=timezone.utc)
        pending = [c for c in conversations if extracted.get(c["id"], {}).get("source_hash") != c["hash"]]
        gaps = [(a, b) for a, b in zip(created, created[1:]) if days_between(a, b) > QUIET_GAP_DAYS]
        row.update({
            "archive": path.name, "archived": archived.date().isoformat(),
            "conversations": len(conversations),
            "first": created[0].date().isoformat() if created else "-",
            "newest_activity": ingest.iso(newest) if newest else "-",
            "newest_title": newest_conv["title"] if newest_conv else "-",
            "export_age_days": days_between(archived, now),
            "silent_before_export_days": days_between(newest, archived) if newest else None,
            "pending_extraction": len(pending),
            "memories": memories[(source, "total")], "approved": memories[(source, "approved")],
            "published": memories[(source, "published")],
            "quiet_periods": [f"{a.date()} to {b.date()} ({days_between(a, b)} days)" for a, b in gaps],
        })
        if row["export_age_days"] is not None and row["export_age_days"] > STALE_DAYS:
            row["warnings"].append(f"Export is {row['export_age_days']} days old. Request a new one: {CAPTURE[source]}.")
        if row["silent_before_export_days"] and row["silent_before_export_days"] > STALE_DAYS:
            row["warnings"].append(f"No activity in the {row['silent_before_export_days']} days before the export. "
                                   "Either unused, or this export misses where you chat now.")
        if pending:
            row["warnings"].append(f"{len(pending)} conversations not yet extracted. Run sync.py.")
        row["warnings"].append(UNIMPORTED[source])
        rows.append(row)
    return rows


def render(rows, now=None):
    now = now or datetime.now(timezone.utc)
    out = ["# Sync Status", "", f"Generated {ingest.iso(now)} by status.py; rerunning overwrites this note.", "",
           "| Provider | Conversations | First | Newest activity | Export archived | Age (days) | "
           "Pending extraction | Memories | Approved | Published |",
           "|---|---:|---|---|---|---:|---:|---:|---:|---:|"]
    for r in rows:
        if "conversations" not in r:
            out.append(f"| {r['source']} | 0 | - | - | - | - | - | - | - | - |")
            continue
        out.append(f"| {r['source']} | {r['conversations']} | {r['first']} | {r['newest_activity'][:10]} | "
                   f"{r['archived']} | {r['export_age_days']} | {r['pending_extraction']} | {r['memories']} | "
                   f"{r['approved']} | {r['published']} |")
    out += ["", "## Gaps and actions", ""]
    for r in rows:
        out.append(f"### {r['source']}")
        if r.get("newest_title"):
            out.append(f"- Last captured conversation: {policy.cell(r['newest_title'])} ({r['newest_activity']})")
        out += [f"- {w}" for w in r["warnings"]]
        if r.get("quiet_periods"):
            out.append(f"- Quiet periods over {QUIET_GAP_DAYS} days: " + "; ".join(r["quiet_periods"]))
        out.append(f"- How to export: {CAPTURE[r['source']]}")
        out.append("")
    return "\n".join(out) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--vault", type=Path, default=Path.home() / "personal_chats")
    args = parser.parse_args()
    rows = provider_status(args.vault)
    report = render(rows)
    policy.write_report(args.vault, "Sync Status.md", report)
    print(report)
    return 0


if __name__ == "__main__":
    sys.exit(main())
