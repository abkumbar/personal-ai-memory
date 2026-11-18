#!/usr/bin/env python3
"""Idempotently import Claude, Grok and ChatGPT exports into the Obsidian vault.

Guarantees:
- One note per conversation, found by `source_id` so renamed or moved notes still match.
- Existing properties are never changed, except import-owned ones when an export changes.
- A note body is only rewritten if it still matches what the importer last wrote.
  Otherwise the new render goes to state/conflicts/ and the note is left alone.
- Every conversation defaults to `access: private`. `grok_approved` is never read or set.

Standard library only.
"""

import argparse
from collections import Counter
from datetime import datetime, timedelta, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import sys
import tempfile

EPOCH = datetime(1970, 1, 1, tzinfo=timezone.utc)
STATE_DIR = Path(__file__).resolve().parent / "state"

# Render 1 is the transcript format of the original importer (visible text only).
# Render 2 adds attachments, thinking, tool calls and other export-only content.
LEGACY_RENDER = 1
FULL_RENDER = 2

OWNED_KEYS = ("updated_at_utc", "updated", "message_count", "source_file")
GROK_CORE_FIELDS = {
    "_id", "conversation_id", "message", "sender", "create_time", "model",
    "parent_response_id", "xai_user_id", "metadata",
}
ROLES = {"human": "User", "user": "User", "assistant": "Assistant",
         "system": "System", "tool": "Tool"}


def parse_time(value):
    if value is None or value == "":
        return None
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return EPOCH + timedelta(seconds=value)
    if isinstance(value, dict):
        value = value["$date"]
        if isinstance(value, dict):
            value = int(value["$numberLong"])
        if isinstance(value, int):
            return EPOCH + timedelta(milliseconds=value)
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def iso(value):
    return value.isoformat().replace("+00:00", "Z") if value else None


def scalar(value):
    return (json.dumps(value, ensure_ascii=False)
            .replace("\u0085", "\\u0085").replace("\u2028", "\\u2028")
            .replace("\u2029", "\\u2029").replace("\ufeff", "\\uFEFF"))


def sha(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def fenced(body, info="text"):
    longest = max((len(run) for run in re.findall(r"`+", body)), default=0)
    fence = "`" * max(3, longest + 1)
    return f"{fence}{info}\n{body}\n{fence}"


def as_text(value):
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True)


# --- Reading exports -------------------------------------------------------

FIELDS = {
    "claude": ("uuid", "name", "created_at", "updated_at"),
    "grok": ("id", "title", "create_time", "modify_time"),
    "chatgpt": ("conversation_id", "title", "create_time", "update_time"),
}


def load_json(path):
    with path.open(encoding="utf-8-sig") as file:
        return json.load(file)


def chatgpt_shards(path):
    """A ChatGPT export is a folder of conversations-NNN.json shards, or one such file."""
    return sorted(path.glob("conversations*.json")) if path.is_dir() else [path]


def chatgpt_text(message):
    content = message["content"]
    if content.get("content_type") not in ("text", "multimodal_text"):
        return ""
    parts = []
    for part in content.get("parts") or []:
        if isinstance(part, str):
            parts.append(part)
        elif isinstance(part, dict) and part.get("content_type") == "audio_transcription":
            parts.append(part.get("text") or "")  # The user's own spoken words.
    return "\n\n".join(part for part in parts if part.strip())


def chatgpt_messages(chat):
    """Follow the active branch from current_node back to the root."""
    mapping = chat["mapping"]
    chain, node_id, seen = [], chat.get("current_node"), set()
    while node_id in mapping and node_id not in seen:
        seen.add(node_id)
        chain.append(mapping[node_id])
        node_id = mapping[node_id].get("parent")
    if not chain:
        chain = sorted(mapping.values(), key=lambda n: (n.get("message") or {}).get("create_time") or 0,
                       reverse=True)
    messages = []
    for node in reversed(chain):
        message = node.get("message")
        if not message:
            continue
        metadata = message.get("metadata") or {}
        if metadata.get("is_visually_hidden_from_conversation"):
            continue
        role, text = message["author"]["role"], chatgpt_text(message)
        if role == "system" and not text.strip():
            continue
        messages.append({"_id": message["id"], "parent_response_id": node.get("parent"),
                         "create_time": message.get("create_time"), "sender": role,
                         "model": metadata.get("model_slug"), "message": text, "raw": message})
    return messages


def read_export(source, path):
    conversations = []
    if source == "chatgpt":
        for shard in chatgpt_shards(path):
            for chat in load_json(shard):
                conversations.append(make_conversation(source, chat, chatgpt_messages(chat)))
        return conversations
    data = load_json(path)
    if source == "claude":
        for chat in data:
            conversations.append(make_conversation(source, chat, chat["chat_messages"]))
    else:
        for entry in data["conversations"]:
            messages = [item["response"] for item in entry["responses"]]
            conversations.append(make_conversation(source, entry["conversation"], messages))
    return conversations


def make_conversation(source, chat, messages):
    id_key, title_key, created_key, updated_key = FIELDS[source]
    chat_id = chat[id_key]
    if not isinstance(chat_id, str) or not chat_id.strip():
        raise ValueError("Empty conversation ID.")
    return {
        "source": source,
        "id": chat_id,
        "title": chat[title_key] or "Untitled",
        "created": parse_time(chat.get(created_key)),
        "updated": parse_time(chat.get(updated_key)),
        "messages": messages,
        "hash": sha(json.dumps([chat, messages], sort_keys=True, ensure_ascii=False)),
        "filename": f"{source}-{sha(chat_id)}.md",
    }


# --- Rendering -------------------------------------------------------------

def claude_text(message):
    blocks = message.get("content") or []
    parts = [block["text"] for block in blocks if block.get("type") == "text"]
    if parts:
        return "\n\n".join(parts)
    return "" if blocks else (message.get("text") or "")


def claude_extras(message):
    extras = []
    for item in message.get("attachments") or []:
        extras.append((f"Attachment: {item.get('file_name') or 'unnamed'}",
                       item.get("extracted_content") or ""))
    for item in message.get("files") or []:
        extras.append((f"File reference: {item.get('file_name') or 'unnamed'}",
                       item.get("file_uuid") or ""))
    for block in message.get("content") or []:
        kind = block.get("type")
        if kind == "thinking":
            extras.append(("AI thinking", block.get("thinking") or ""))
        elif kind == "tool_use":
            extras.append((f"Tool call: {block.get('name')}", block.get("input")))
        elif kind == "tool_result":
            extras.append((f"Tool result: {block.get('name')}", block.get("content")))
        elif kind == "injected_prompt_block":
            extras.append(("Injected prompt (not written by the user)", block.get("prompt") or ""))
    return extras


def chatgpt_extras(message):
    raw = message["raw"]
    content, metadata = raw["content"], raw.get("metadata") or {}
    kind = content.get("content_type")
    extras = []
    if kind == "thoughts":
        extras.append(("AI thinking", content.get("thoughts")))
    elif kind == "reasoning_recap":
        extras.append(("AI reasoning recap", content.get("content")))
    elif kind not in ("text", "multimodal_text"):
        extras.append((f"Export content: {kind}", content))
    for part in content.get("parts") or []:
        if isinstance(part, dict) and part.get("content_type") != "audio_transcription":
            extras.append((f"Media: {part.get('content_type')}", part.get("asset_pointer") or part))
    for item in metadata.get("attachments") or []:
        extras.append((f"Attachment: {item.get('name') or 'unnamed'}", item))
    return extras


def grok_extras(message):
    return [(f"Export field: {key}", value) for key, value in sorted(message.items())
            if key not in GROK_CORE_FIELDS and value not in (None, "", [], {})]


def render_message(source, message, render):
    claude = source == "claude"
    sender = message["sender"]
    body = claude_text(message) if claude else message["message"]
    fields = [
        ("Message ID", message.get("uuid" if claude else "_id")),
        ("Parent ID", message.get("parent_message_uuid" if claude else "parent_response_id")),
        ("Sent (UTC)", iso(parse_time(message.get("created_at" if claude else "create_time")))),
        ("Original sender", sender),
        ("Model", message.get("model")),
    ]
    metadata = " · ".join(f"{key} {scalar(value)}" for key, value in fields if value)
    delimiter = "`" * (1 + max((len(run) for run in re.findall(r"`+", metadata)), default=0))
    section = [f"## {ROLES.get(sender.casefold(), 'Message')}",
               f"{delimiter} {metadata} {delimiter}",
               fenced(body) if body else "*No transcript text in this message.*"]
    if render == LEGACY_RENDER:
        count = sum(len(message.get(key) or [])
                    for key in (("attachments", "files") if claude else ("file_attachments",)))
        if count:
            section.append(f"*Attachment entries in the export: {count}.*")
    else:
        extras = (claude_extras if claude else
                  chatgpt_extras if source == "chatgpt" else grok_extras)(message)
        for label, value in extras:
            section.append(f"### {label}\n\n{fenced(as_text(value))}")
    return "\n\n".join(section)


def render_body(conversation, render):
    return "\n\n".join(render_message(conversation["source"], message, render)
                       for message in conversation["messages"])


def date_of(value):
    return value.date().isoformat() if value else None


def default_properties(conversation):
    """Properties added when absent. Raw YAML values, keyed by property name."""
    defaults = {
        "id": scalar(f"{conversation['source']}-{conversation['id']}"),
        "type": "conversation",
        "created": date_of(conversation["created"]),
        "updated": date_of(conversation["updated"]),
        "project": "",
        "topics": "[]",
        "sensitivity": "private",
        "access": "private",
    }
    return {key: value for key, value in defaults.items() if value is not None}


def owned_properties(conversation, source_file):
    values = {
        "updated_at_utc": scalar(iso(conversation["updated"])) if conversation["updated"] else None,
        "updated": date_of(conversation["updated"]),
        "message_count": str(len(conversation["messages"])),
        "source_file": scalar(source_file),
    }
    return {key: values[key] for key in OWNED_KEYS if values[key] is not None}


def new_note(conversation, source_file):
    created = conversation["created"]
    lines = ["---"]
    if created:
        lines.append(f"date: {date_of(created)}")
    lines += [f"title: {scalar(conversation['title'])}",
              f"source: {conversation['source']}",
              f"source_id: {scalar(conversation['id'])}",
              f"source_file: {scalar(source_file)}"]
    if created:
        lines.append(f"created_at_utc: {scalar(iso(created))}")
    if conversation["updated"]:
        lines.append(f"updated_at_utc: {scalar(iso(conversation['updated']))}")
    lines += [f"message_count: {len(conversation['messages'])}",
              "privacy: private", "grok_approved: false", "publish: false"]
    for key, value in default_properties(conversation).items():
        lines.append(f"{key}: {value}".rstrip())
    lines.append("---")
    return "\n".join(lines) + "\n\n" + render_body(conversation, FULL_RENDER) + "\n"


# --- Editing notes ---------------------------------------------------------

def split_note(text):
    """Return (frontmatter lines, body). Frontmatter is None if the note has none."""
    if not text.startswith("---\n"):
        return None, text
    end = text.find("\n---", 3)
    if end == -1 or text[end + 4:end + 5] not in ("", "\n"):
        return None, text
    return text[4:end].split("\n"), text[end + 5:]


def top_level_keys(lines):
    return {match.group(1) for line in lines
            if (match := re.match(r"([^\s:#-][^:]*):(\s|$)", line))}


def add_missing(lines, values):
    present = top_level_keys(lines)
    return lines + [f"{key}: {value}".rstrip() for key, value in values.items()
                    if key not in present]


def set_keys(lines, values):
    result = list(lines)
    for key, value in values.items():
        pattern = re.compile(rf"{re.escape(key)}:(\s|$)")
        for index, line in enumerate(result):
            if pattern.match(line):
                result[index] = f"{key}: {value}"
                break
        else:
            result.append(f"{key}: {value}")
    return result


def join_note(lines, body):
    frontmatter = "\n".join(lines)
    return f"---\n{frontmatter}\n---\n{body}" if frontmatter else f"---\n---\n{body}"


def normalized(body):
    return body.strip("\n")


def write_atomic(path, content):
    descriptor, temporary = tempfile.mkstemp(prefix=".ai-import-", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as file:
            file.write(content)
        if path.exists():
            shutil.copymode(path, temporary)
        os.replace(temporary, path)
    except BaseException:
        if os.path.exists(temporary):
            os.unlink(temporary)
        raise


def index_notes(folder):
    found = {}
    for path in folder.rglob("*.md"):
        lines, _ = split_note(path.read_text(encoding="utf-8"))
        for line in lines or []:
            match = re.match(r"source_id:\s*(.+?)\s*$", line)
            if match:
                found[match.group(1).strip("\"'")] = path
                break
    return found


# --- Import ----------------------------------------------------------------

def archive_export(path, source, vault, dry_run):
    """Copy the export into 01-Raw-Exports once, read-only. Folders keep only conversation shards."""
    shards = chatgpt_shards(path) if path.is_dir() else [path]
    hasher = hashlib.sha256()
    for shard in shards:
        hasher.update(shard.name.encode("utf-8") + b"\0" + shard.read_bytes())
    digest = hasher.hexdigest() if path.is_dir() else hashlib.sha256(path.read_bytes()).hexdigest()
    folder = vault / "01-Raw-Exports" / source
    existing = list(folder.glob(f"*-{digest[:16]}-*")) if folder.exists() else []
    if existing:
        return "already archived"
    if dry_run:
        return "would archive"
    folder.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d")
    target = folder / f"{stamp}-{digest[:16]}-{path.name}"
    if path.is_dir():
        target.mkdir(mode=0o700)
        for shard in shards:
            shutil.copyfile(shard, target / shard.name)
            (target / shard.name).chmod(0o400)
    else:
        shutil.copyfile(path, target)
        target.chmod(0o400)
    return "archived"


def write_conflict(state_dir, conversation, body):
    folder = state_dir / "conflicts"
    folder.mkdir(parents=True, exist_ok=True)
    stem = Path(conversation["filename"]).stem
    path = folder / f"{stem}.{conversation['hash'][:12]}.md"
    path.write_text(body + "\n", encoding="utf-8")
    path.chmod(0o600)


def sync_conversation(conversation, source_file, folder, notes, manifest, state_dir,
                      upgrade_bodies, dry_run):
    chat_id = conversation["id"]
    entry = manifest.get(chat_id)
    path = notes.get(chat_id)

    if path is None:
        if entry:
            return "missing (deleted by user, not recreated)"
        content = new_note(conversation, source_file)
        if not dry_run:
            write_atomic(folder / conversation["filename"], content)
            manifest[chat_id] = {"source_hash": conversation["hash"], "render": FULL_RENDER,
                                 "body_hash": sha(normalized(render_body(conversation, FULL_RENDER)))}
        return "created"

    original = path.read_text(encoding="utf-8")
    lines, body = split_note(original)
    if lines is None:
        return "conflict (no frontmatter)"
    body_hash = sha(normalized(body))

    if entry is None:
        render = next((r for r in (LEGACY_RENDER, FULL_RENDER)
                       if normalized(render_body(conversation, r)) == normalized(body)), None)
        entry = {"source_hash": conversation["hash"] if render else None,
                 "render": render, "body_hash": body_hash if render else None}

    wants_render = (entry["source_hash"] != conversation["hash"]
                    or (upgrade_bodies and entry["render"] != FULL_RENDER))
    status = "unchanged"
    updated_lines = add_missing(lines, default_properties(conversation))
    updated_body = body

    if wants_render:
        full = render_body(conversation, FULL_RENDER)
        if entry["body_hash"] == body_hash:
            updated_body = "\n" + full + "\n"
            updated_lines = set_keys(updated_lines, owned_properties(conversation, source_file))
            entry = {"source_hash": conversation["hash"], "render": FULL_RENDER,
                     "body_hash": sha(normalized(full))}
            status = "updated"
        else:
            status = "conflict (note edited, new version saved to state/conflicts)"
            if not dry_run:
                write_conflict(state_dir, conversation, full)

    result = join_note(updated_lines, updated_body)
    if status == "unchanged" and result != original:
        status = "properties added"
    if not dry_run:
        if result != original:
            write_atomic(path, result)
        manifest[chat_id] = entry
    return status


def load_manifest(path):
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def run(exports, vault, state_dir=STATE_DIR, upgrade_bodies=False, dry_run=False):
    folder = vault / "02-Conversations"
    if not folder.is_dir():
        raise SystemExit(f"Not a vault with 02-Conversations: {vault}")

    # Parse everything first so a malformed export writes nothing.
    parsed = [(source, path, read_export(source, path)) for source, path in exports]
    ids = [c["id"] for _, _, conversations in parsed for c in conversations]
    if len(ids) != len(set(ids)):
        raise SystemExit("Duplicate conversation IDs across exports. Nothing was written.")

    manifest_path = state_dir / "manifest.json"
    manifest = load_manifest(manifest_path)
    notes = index_notes(folder)
    counts = Counter()
    conflicts = []
    if not dry_run:
        state_dir.mkdir(parents=True, exist_ok=True)
        state_dir.chmod(0o700)

    try:
        for source, path, conversations in parsed:
            print(f"{source} raw export: {archive_export(path, source, vault, dry_run)}")
            for conversation in conversations:
                status = sync_conversation(conversation, path.name, folder, notes, manifest,
                                           state_dir, upgrade_bodies, dry_run)
                counts[source, status] += 1
                if status.startswith("conflict"):
                    conflicts.append(f"{conversation['filename']}: {status}")
    finally:
        if not dry_run:
            write_atomic_manifest(manifest_path, manifest)

    prefix = "[dry run] " if dry_run else ""
    for (source, status), count in sorted(counts.items()):
        print(f"{prefix}{source}: {count} {status}")
    for line in conflicts:
        print(f"CONFLICT {line}", file=sys.stderr)
    return counts


def write_atomic_manifest(path, manifest):
    write_atomic(path, json.dumps(manifest, indent=1, sort_keys=True) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--claude", type=Path, help="path to Claude conversations.json")
    parser.add_argument("--grok", type=Path, help="path to prod-grok-backend.json")
    parser.add_argument("--chatgpt", type=Path,
                        help="ChatGPT export folder (conversations-NNN.json) or one conversations file")
    parser.add_argument("--vault", type=Path, default=Path.home() / "personal_chats")
    parser.add_argument("--upgrade-bodies", action="store_true",
                        help="rewrite unedited notes to include attachments, thinking and tool content")
    parser.add_argument("--dry-run", action="store_true", help="report without writing")
    args = parser.parse_args()
    exports = [(s, p) for s, p in (("claude", args.claude), ("grok", args.grok),
                                    ("chatgpt", args.chatgpt)) if p]
    if not exports:
        parser.error("provide at least one of --claude, --grok, --chatgpt")
    run(exports, args.vault, upgrade_bodies=args.upgrade_bodies, dry_run=args.dry_run)
    return 0


if __name__ == "__main__":
    sys.exit(main())
