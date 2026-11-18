#!/usr/bin/env python3
"""Import saved memories and custom instructions as memory notes.

  python3 import_profile.py --claude-memories PATH/memories/<uuid>.json
  python3 import_profile.py                     also reads the ChatGPT paste-in notes

Sources:
- Claude saved memory (from the Claude export's memories/*.json): every bullet of each memory
  file and every paragraph of the "conversations memory" summary becomes one note.
  Attribution `provider_memory`: written by Claude from your chats, not your own words.
- ChatGPT (not in its export): paste into 04-Profile/ChatGPT saved memories.md and
  04-Profile/ChatGPT custom instructions.md. Saved memories are `provider_memory`;
  custom instructions are `user_stated`.

Every note: type memory, category profile, access private. Notes are created once and never
rewritten. When an item no longer appears in its source, its note gets `in_source: false`.
The raw Claude memory file is archived read-only under 01-Raw-Exports/claude-memory/.

Standard library only.
"""

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import shutil
import sys

import ingest

PROFILE_DIR = "04-Profile"
CHATGPT_MEMORIES = "ChatGPT saved memories.md"
CHATGPT_INSTRUCTIONS = "ChatGPT custom instructions.md"
STATE = ingest.STATE_DIR / "profile_import.json"
TEMPLATE = """---
type: profile-inbox
access: private
---

Paste {what} below, one item per line or bullet. Run `python3 import_profile.py` (or `sync.py`)
to turn each line into a private memory note for review. Editing this note later adds new items;
removed lines mark their memory notes `in_source: false`.

"""


def clean(text):
    text = re.sub(r"^\s*(?:[-*+]|\d+[.)])\s+", "", text)
    text = re.sub(r"\*\*|__|`", "", text)
    return re.sub(r"\s+", " ", text).strip()


def body_of(markdown):
    """Drop YAML frontmatter from a memory file."""
    if markdown.startswith("---"):
        end = markdown.find("\n---", 3)
        if end != -1:
            return markdown[end + 4:]
    return markdown


def bullets(markdown):
    items, current = [], None
    for line in body_of(markdown).splitlines():
        if re.match(r"^(?:[-*+]|\d+[.)])\s+", line):
            if current:
                items.append(current)
            current = clean(line)
        elif current is not None and line.startswith((" ", "\t")) and line.strip():
            current += " " + clean(line)  # Continuation or sub-bullet of the current item.
        elif line.strip() and not line.startswith("#"):
            if current:  # A plain line is an item of its own.
                items.append(current)
                current = None
            items.append(clean(line))
    if current:
        items.append(current)
    return [i for i in items if len(i) >= 3]


def paragraphs(text):
    blocks = [clean(b) for b in re.split(r"\n\s*\n", text) if b.strip()]
    merged = []
    for block in blocks:
        if merged and len(merged[-1]) < 60 and not merged[-1].endswith((".", "!", "?")):
            merged[-1] = f"{merged[-1]}: {block}"  # A short heading-like line joins what follows.
        else:
            merged.append(block)
    return merged


def claude_items(path):
    data = json.loads(path.read_text(encoding="utf-8"))
    items = []
    for memory_file in data.get("memory_files", []):
        rel = memory_file["path"].lstrip("/")
        area = Path(rel).stem
        kind = rel.split("/")[0] if "/" in rel else "profile"
        for text in bullets(memory_file["content"]):
            items.append({"source": "claude", "origin": f"Claude saved memory ({rel})", "text": text,
                          "attribution": "provider_memory", "updated": memory_file.get("updated_at"),
                          "project": area if kind == "areas" else "", "subject": area})
    for text in paragraphs(data.get("conversations_memory", "")):
        items.append({"source": "claude", "origin": "Claude saved memory (conversations summary)", "text": text,
                      "attribution": "provider_memory", "updated": None, "project": "", "subject": "claude-summary"})
    return items


def chatgpt_items(vault):
    items = []
    for name, attribution, subject in ((CHATGPT_MEMORIES, "provider_memory", "chatgpt-memory"),
                                       (CHATGPT_INSTRUCTIONS, "user_stated", "chatgpt-instructions")):
        path = vault / PROFILE_DIR / name
        if not path.exists():
            continue
        text = body_of(path.read_text(encoding="utf-8"))
        text = text.split("\n\n", 2)[-1] if text.lstrip().startswith("Paste ") else text
        for line in bullets(text):
            if line.startswith("Paste ") or "import_profile.py" in line:
                continue
            items.append({"source": "chatgpt", "origin": f"ChatGPT ({path.stem})", "text": line,
                          "attribution": attribution, "updated": None, "project": "", "subject": subject})
    return items


def memory_id(item):
    key = f"{item['source']}|{item['subject']}|{re.sub(r'[^a-z0-9]+', ' ', item['text'].casefold()).strip()}"
    return "mem-" + hashlib.sha256(key.encode("utf-8")).hexdigest()[:16]


def render(item, memory_id_, now):
    scalar = ingest.scalar
    title = item["text"] if len(item["text"]) <= 90 else item["text"][:87].rsplit(" ", 1)[0] + "..."
    updated = ingest.parse_time(item["updated"]) if item["updated"] else None
    lines = ["---", f"id: {memory_id_}", "type: memory", f"title: {scalar(title)}", "category: profile",
             f"attribution: {item['attribution']}", "status: active", f"subject: {scalar(item['subject'])}",
             f"project: {scalar(item['project'])}" if item["project"] else "project:", "topics: []", "flags: []",
             "revises:", f"source: {item['source']}",
             f"source_conversation_id: {scalar(item['source'] + '-profile')}", "source_message_ids: []"]
    if updated:
        lines.append(f"created: {updated.date().isoformat()}")
    lines += [f"extracted_at: {scalar(now)}", 'extractor_model: "profile-import"', "extractor_prompt_version: 1",
              "in_source: true", "sensitivity: private", "access: private", "---"]
    who = "You wrote this" if item["attribution"] == "user_stated" else "The provider recorded this about you"
    return "\n".join(lines) + f"\n\n{item['text']}\n\n## Evidence\n\n{who}: {item['origin']}.\n"


def ensure_templates(vault):
    folder = vault / PROFILE_DIR
    folder.mkdir(exist_ok=True)
    for name, what in ((CHATGPT_MEMORIES, "your ChatGPT saved memories (Settings > Personalization > Memory > Manage)"),
                       (CHATGPT_INSTRUCTIONS, "your ChatGPT custom instructions (Settings > Personalization > Custom instructions)")):
        path = folder / name
        if not path.exists():
            path.write_text(TEMPLATE.format(what=what), encoding="utf-8")


def archive(path, vault):
    digest = hashlib.sha256(path.read_bytes()).hexdigest()[:16]
    folder = vault / "01-Raw-Exports" / "claude-memory"
    if not list(folder.glob(f"*-{digest}-*")) if folder.exists() else True:
        folder.mkdir(parents=True, exist_ok=True)
        target = folder / f"{datetime.now(timezone.utc):%Y%m%d}-{digest}-{path.name}"
        shutil.copyfile(path, target)
        target.chmod(0o400)


def run(vault, claude_memories=None, state_path=STATE, dry_run=False):
    now = ingest.iso(datetime.now(timezone.utc))
    items, sources = [], set()
    if claude_memories:
        items += claude_items(claude_memories)
        sources.add("claude")
    if not dry_run:
        ensure_templates(vault)
    chatgpt = chatgpt_items(vault)
    items += chatgpt
    sources.add("chatgpt")  # The paste-in notes always define the current ChatGPT set.
    state = json.loads(state_path.read_text(encoding="utf-8")) if state_path.exists() else {}
    folder = vault / "03-Memories"
    counts = {"created": 0, "existing": 0, "no longer in source": 0, "back in source": 0}
    current = {}
    for item in items:
        current.setdefault(item["source"], set()).add(memory_id(item))
    for item in items:
        mid = memory_id(item)
        path = folder / f"{mid}.md"
        if path.exists():
            counts["existing"] += 1
            continue
        counts["created"] += 1
        if not dry_run:
            ingest.write_atomic(path, render(item, mid, now))
    for source in sources:
        for mid in set(state.get(source, [])) | current.get(source, set()):
            path = folder / f"{mid}.md"
            if not path.exists():
                continue
            text = path.read_text(encoding="utf-8")
            present = mid in current.get(source, set())
            new = re.sub(r"^in_source: (true|false)$", f"in_source: {'true' if present else 'false'}", text, count=1, flags=re.M)
            if new != text:
                counts["back in source" if present else "no longer in source"] += 1
                if not dry_run:
                    ingest.write_atomic(path, new)
        state[source] = sorted(current.get(source, set()) | set(state.get(source, [])))
    if not dry_run:
        if claude_memories:
            archive(claude_memories, vault)
        state_path.parent.mkdir(parents=True, exist_ok=True)
        ingest.write_atomic(state_path, json.dumps(state, indent=1) + "\n")
    by_source = {s: len(v) for s, v in current.items()}
    print(f"{'[dry run] ' if dry_run else ''}items by source: {by_source}; " +
          ", ".join(f"{k}: {v}" for k, v in counts.items()))
    return counts


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--vault", type=Path, default=Path.home() / "personal_chats")
    parser.add_argument("--claude-memories", type=Path, help="memories/<uuid>.json from a Claude export")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    run(args.vault, args.claude_memories, dry_run=args.dry_run)
    return 0


if __name__ == "__main__":
    sys.exit(main())
