#!/usr/bin/env python3
"""Publish approved memory notes to the GrokBot Context directory.

  python3 publish.py sync [--dry-run]   publish, update and revoke once
  python3 publish.py watch              keep running and sync whenever memories change

Rules:
- Only notes with `type: memory` and `access: approved` are published. Raw conversations never are.
- Approval binds to content. `approved_hash` is a fingerprint of the approved title and body.
  Policy approvals carry one from policy.py. A manual approval gets one stamped the first
  time it is seen. If the note is edited afterwards, the published copy stays at the approved
  version and the note shows `publish_status: held (edited after approval)`. To approve the
  new text, delete `approved_hash` (manual) or run policy.py again (policy).
- Anything flagged `credentials`, or whose published text looks like a secret, is held and
  never published, even when approved by hand.
- A note that is no longer approved, or deleted, is removed from the published directory.
- Published copies hold title, statement, category, attribution, project, dates and source IDs.
  Evidence quotes, flags and local links stay in the vault.
- Every publish, update, hold and removal goes to state/publish_audit.jsonl.

Standard library only.
"""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import sys
import time

import flag
import ingest
import policy

PUBLISH_DIR = Path.home() / "GrokBotContext"
INDEX_NOTE = "GrokBot Context.md"
STATUS_KEYS = ("publish_status", "published_at")
WATCH_SECONDS = 10


def now():
    return ingest.iso(datetime.now(timezone.utc))


def statement_of(body):
    return body.split("\n## Evidence", 1)[0].strip()


def record_for(lines, body, project, approved_hash):
    return {
        "id": policy.value(lines, "id"),
        "title": policy.value(lines, "title"),
        "statement": statement_of(body),
        "category": policy.value(lines, "category"),
        "attribution": policy.value(lines, "attribution"),
        "project": project,
        "created": policy.value(lines, "created"),
        "source": policy.value(lines, "source"),
        "source_conversation_id": policy.value(lines, "source_conversation_id"),
        "source_message_ids": flag.read_list(lines, "source_message_ids")[0],
        "approved_by": policy.value(lines, "approved_by") or "manual",
        "approved_hash": approved_hash,
    }


def render_published(record):
    keys = ("id", "title", "category", "attribution", "project", "created", "source",
            "source_conversation_id", "approved_by", "approved_hash")
    lines = ["---"] + [f"{k}: {ingest.scalar(record[k])}" for k in keys if record[k]]
    lines.append(f"source_message_ids: {json.dumps(record['source_message_ids'])}")
    said_by = {"ai_suggested": "AI suggestion",
               "provider_memory": "Recorded by the provider from the user's chats"}.get(
        record["attribution"], "Stated by the user")
    return "\n".join(lines + ["---", "", f"# {record['title']}", "",
                              record["statement"], "", f"*{said_by}.*", ""])


def looks_like_secret(text, keyword_rules):
    return "credentials" in flag.match_flags(text, {"credentials": keyword_rules["credentials"]})


class Publisher:
    def __init__(self, vault, publish_dir=PUBLISH_DIR, state_dir=ingest.STATE_DIR, policy_path=policy.POLICY_PATH):
        self.vault = vault
        self.policy_path = policy_path
        self.publish_dir = publish_dir
        self.state_dir = state_dir
        self.state_path = state_dir / "published.json"
        self.holds_path = state_dir / "publish_holds.json"
        self.keyword_rules = flag.load_rules()

    def load_state(self):
        return json.loads(self.state_path.read_text(encoding="utf-8")) if self.state_path.exists() else {}

    def sync(self, dry_run=False):
        published = self.load_state()
        holds = json.loads(self.holds_path.read_text(encoding="utf-8")) if self.holds_path.exists() else {}
        conv_projects = policy.conversation_projects(self.vault)
        rules = policy.load_policy(self.policy_path) if self.policy_path.exists() else None
        counts, audit, seen, note_writes = {}, [], set(), []
        current = {}

        def count(key):
            counts[key] = counts.get(key, 0) + 1

        def log(event, memory_id, content_hash, reason=""):
            audit.append({"at": now(), "event": event, "memory": memory_id,
                          "content_hash": content_hash, "reason": reason})

        for path, text in policy.memory_notes(self.vault):
            lines, body = ingest.split_note(text)
            memory_id = policy.value(lines, "id")
            seen.add(memory_id)
            access = policy.value(lines, "access")
            status, extra = None, {}
            if access != "approved":
                holds.pop(memory_id, None)
                if memory_id in published:
                    count("removed (no longer approved)")
                    log("unpublish", memory_id, published[memory_id]["hash"], "not approved")
                    published.pop(memory_id)
                if any(policy.value(lines, k) for k in STATUS_KEYS):
                    note_writes.append((path, text, ingest.join_note(policy.remove_keys(lines, STATUS_KEYS), body)))
                continue

            content_hash = policy.content_hash(lines, body)
            approved_hash = policy.value(lines, "approved_hash")
            if not approved_hash:
                if policy.value(lines, "approved_by").startswith("policy:"):
                    status = "held (policy approval missing fingerprint)"
                else:
                    approved_hash = content_hash
                    extra = {"approved_hash": approved_hash, "approved_by": ingest.scalar("manual"),
                             "approved_at": ingest.scalar(now())}
                    count("manual approvals fingerprinted")
            project = policy.read_memory(lines, body, conv_projects)["project"]
            if rules:  # Publish the alias group's name so a project reads as one.
                project = policy.display_project(rules, project)
            record = record_for(lines, body, project, approved_hash)
            flags = set(flag.read_list(lines, "flags")[0])
            secret = "credentials" in flags or looks_like_secret(record["title"] + "\n" + record["statement"],
                                                                  self.keyword_rules)
            if status:
                pass
            elif secret:
                status = "held (credentials)"
                if memory_id in published:
                    log("unpublish", memory_id, published.pop(memory_id)["hash"], "credentials")
                    count("removed (credentials)")
            elif approved_hash != content_hash:
                status = "held (edited after approval)"
                if memory_id in published:
                    current[memory_id] = published[memory_id]["record"]
            else:
                status = "published"
                previous = published.get(memory_id)
                if previous is None or previous["record"] != record:
                    event = "publish" if previous is None else "update"
                    count("published" if previous is None else "updated")
                    log(event, memory_id, content_hash)
                    published[memory_id] = {"hash": content_hash, "record": record, "at": now()}
                current[memory_id] = published[memory_id]["record"]
            if status != "published":
                count(status)
                if holds.get(memory_id) != status:
                    log("hold", memory_id, content_hash, status)
                holds[memory_id] = status
            else:
                holds.pop(memory_id, None)

            new_lines = ingest.set_keys(lines, dict(extra, publish_status=ingest.scalar(status)))
            if status == "published":
                new_lines = ingest.set_keys(new_lines, {"published_at": ingest.scalar(published[memory_id]["at"])})
            new_text = ingest.join_note(new_lines, body)
            if new_text != text:
                note_writes.append((path, text, new_text))

        holds = {m: r for m, r in holds.items() if m in seen}
        for memory_id in [m for m in published if m not in seen]:
            count("removed (note deleted)")
            log("unpublish", memory_id, published.pop(memory_id)["hash"], "note deleted")

        if not dry_run:
            self.write_outputs(current, published)
            for path, old, new in note_writes:
                if path.read_text(encoding="utf-8") == old:  # Skip if edited while we worked.
                    ingest.write_atomic(path, new)
            ingest.write_atomic(self.holds_path, json.dumps(holds, indent=1, sort_keys=True) + "\n")
            if audit:
                with (self.state_dir / "publish_audit.jsonl").open("a", encoding="utf-8") as log_file:
                    for entry in audit:
                        log_file.write(json.dumps(entry) + "\n")
        counts["live"] = len(current)
        return counts

    def write_outputs(self, current, published):
        self.state_dir.mkdir(parents=True, exist_ok=True)
        folder = self.publish_dir / "memories"
        folder.mkdir(parents=True, exist_ok=True)
        self.publish_dir.chmod(0o700)
        wanted = {f"{memory_id}.md": render_published(record) for memory_id, record in current.items()}
        for path in folder.glob("*.md"):
            if path.name not in wanted:
                path.unlink()
        for name, content in wanted.items():
            path = folder / name
            if not path.exists() or path.read_text(encoding="utf-8") != content:
                ingest.write_atomic(path, content)
                path.chmod(0o600)
        records = sorted(current.values(), key=lambda r: (r["project"].casefold(), r["created"], r["id"]))
        ingest.write_atomic(self.publish_dir / "memories.jsonl",
                            "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records))
        ingest.write_atomic(self.state_path, json.dumps(published, indent=1, sort_keys=True) + "\n")
        self.state_path.chmod(0o600)
        self.write_index(records)

    def write_index(self, records):
        rows = ["# GrokBot Context", "",
                "Generated by publish.py; rerunning overwrites this note. Grok reads the published copies in "
                f"`{self.publish_dir}`, not this vault.", "",
                f"{len(records)} memories published.", "",
                "| Memory | Project | Category | Attribution |", "|---|---|---|---|"]
        rows += [f"| [[{r['id']}\\|{policy.cell(r['title'])}]] | {policy.cell(r['project'])} | "
                 f"{r['category']} | {r['attribution']} |" for r in records]
        path = self.vault / INDEX_NOTE
        content = "\n".join(rows) + "\n"
        if not path.exists() or path.read_text(encoding="utf-8") != content:
            ingest.write_atomic(path, content)


def signature(vault):
    paths = list((vault / "03-Memories").glob("*.md")) + list((vault / "02-Conversations").rglob("*.md"))
    return len(paths), max((p.stat().st_mtime for p in paths), default=0)


def watch(publisher):
    last = None
    print(f"watching {publisher.vault} every {WATCH_SECONDS}s; Ctrl-C to stop", flush=True)
    while True:
        current = signature(publisher.vault)
        if current != last:
            counts = publisher.sync()
            changes = {k: v for k, v in counts.items() if k != "live"}
            if changes or last is None:
                print(f"{now()} {json.dumps(counts, sort_keys=True)}", flush=True)
            last = signature(publisher.vault)  # Include our own note writes.
        time.sleep(WATCH_SECONDS)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("command", choices=["sync", "watch"])
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--vault", type=Path, default=Path.home() / "personal_chats")
    parser.add_argument("--out", type=Path, default=PUBLISH_DIR)
    args = parser.parse_args()
    publisher = Publisher(args.vault, args.out)
    if args.command == "watch":
        try:
            watch(publisher)
        except KeyboardInterrupt:
            return 0
    counts = publisher.sync(dry_run=args.dry_run)
    for key, value in sorted(counts.items()):
        print(f"{'[dry run] ' if args.dry_run else ''}{key}: {value}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
