#!/usr/bin/env python3
"""Run the whole pipeline on the newest exports found in Downloads.

  python3 sync.py                 find exports, import, extract, flag, apply policy, publish, report
  python3 sync.py --dry-run       only show which exports would be used
  python3 sync.py --no-extract    skip the local-model extraction step

Finds exports by content, not name: Claude (conversations.json with chat_messages), ChatGPT
(conversations*.json with mapping, single file or shards), Grok (prod-grok-backend.json).
Zips are read in place and only those JSON files are copied out, into state/inbox/.
Exports already archived are recognized by checksum and skipped by the importer.

Extraction uses the local LM Studio model. If the server or model is not running, this
starts it and stops it again afterwards. Nothing leaves this Mac.

Standard library only.
"""

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import zipfile

import extract
import flag
import import_profile
import ingest
import policy
import publish
import status

DOWNLOADS = Path.home() / "Downloads"
INBOX = ingest.STATE_DIR / "inbox"
LMS = Path.home() / ".lmstudio" / "bin" / "lms"
RUNS = ingest.STATE_DIR / "sync_runs.jsonl"
WANTED = ("conversations.json", "prod-grok-backend.json")


def classify(name, head):
    """Return the provider for a JSON export file from its name and first bytes, or None."""
    base = Path(name).name
    if base == "prod-grok-backend.json":
        return "grok"
    if base.startswith("conversations") and base.endswith(".json"):
        if b'"chat_messages"' in head:
            return "claude"
        if b'"mapping"' in head:
            return "chatgpt"
    return None


def head_of(path, size=262144):
    with open(path, "rb") as file:
        return file.read(size)


def newest_activity(provider, kind, location):
    """Latest conversation timestamp inside an export, so file dates (often 1980 in zips) do not matter."""
    if kind == "zip":
        with zipfile.ZipFile(location) as archive:
            chats = []
            for member in archive.namelist():
                name = Path(member).name
                if classify(name, archive.open(member).read(262144)) == provider:
                    data = json.loads(archive.read(member))
                    chats += [ingest.make_conversation(provider, *pair) for pair in raw_pairs(provider, data)]
    else:
        chats = ingest.read_export(provider, location)
    return max((c["updated"] or c["created"] for c in chats if c["updated"] or c["created"]), default=ingest.EPOCH)


def raw_pairs(provider, data):
    if provider == "claude":
        return [(chat, chat["chat_messages"]) for chat in data]
    if provider == "chatgpt":
        return [(chat, ingest.chatgpt_messages(chat)) for chat in data]
    return [(e["conversation"], [r["response"] for r in e["responses"]]) for e in data["conversations"]]


def find_exports(root=DOWNLOADS, max_depth=5):
    """Return {provider: (newest activity, kind, location)} for the most recent export of each provider."""
    candidates = {}

    def offer(provider, mtime, kind, location):
        if provider and (kind, location) not in [(k, l) for k, l in candidates.get(provider, [])]:
            candidates.setdefault(provider, []).append((kind, location))

    for path in root.rglob("*"):
        if len(path.relative_to(root).parts) > max_depth or INBOX in path.parents:
            continue
        if path.is_file() and path.suffix == ".zip":
            try:
                with zipfile.ZipFile(path) as archive:
                    for member in archive.namelist():
                        if Path(member).name.startswith("conversations") or Path(member).name == WANTED[1]:
                            with archive.open(member) as file:
                                provider = classify(member, file.read(262144))
                            offer(provider, path.stat().st_mtime, "zip", path)
            except (zipfile.BadZipFile, OSError):
                continue
        elif path.is_file() and path.suffix == ".json":
            provider = classify(path.name, head_of(path))
            if provider == "chatgpt" and path.name != "conversations.json":
                offer(provider, path.stat().st_mtime, "dir", path.parent)
            else:
                offer(provider, path.stat().st_mtime, "file", path)
    found = {}
    for provider, options in candidates.items():
        scored = []
        for kind, location in options:
            try:
                scored.append((newest_activity(provider, kind, location), kind, location))
            except (ValueError, KeyError, OSError, zipfile.BadZipFile):
                continue
        if scored:
            found[provider] = max(scored, key=lambda item: item[0])
    return found


def materialize(provider, kind, location):
    """Return a path ingest.py can read. Zips are unpacked (JSON exports only) into state/inbox."""
    if kind != "zip":
        return location
    digest = hashlib.sha256(location.read_bytes()).hexdigest()[:16]
    target = INBOX / f"{provider}-{digest}"
    if not target.exists():
        target.mkdir(parents=True)
        target.chmod(0o700)
        with zipfile.ZipFile(location) as archive:
            for member in archive.namelist():
                name = Path(member).name  # Basename only: no paths from the zip are trusted.
                wanted = classify(name, b'"chat_messages" "mapping"') or name == WANTED[1]
                claude_memory = provider == "claude" and "memories" in Path(member).parts and name.endswith(".json")
                if wanted or claude_memory:
                    with archive.open(member) as source, open(target / name, "wb") as out:
                        shutil.copyfileobj(source, out)
    files = sorted(target.glob("*.json"))
    if provider == "chatgpt" and any(f.name != "conversations.json" for f in files):
        return target
    return next(f for f in files if classify(f.name, head_of(f)) == provider)


def claude_memory_file(location):
    """The Claude saved-memory JSON that came with a Claude export, if any."""
    folder = location if location.is_dir() else location.parent
    for candidate in sorted(folder.glob("*.json")) + sorted((folder / "memories").glob("*.json")):
        if candidate.name == "conversations.json":
            continue
        try:
            if b'"conversations_memory"' in head_of(candidate) or b'"memory_files"' in head_of(candidate):
                return candidate
        except OSError:
            continue
    return None


def pending_extraction(vault, model):
    done = json.loads((ingest.STATE_DIR / "extraction.json").read_text()) \
        if (ingest.STATE_DIR / "extraction.json").exists() else {}
    return sum(1 for source, path in extract.latest_exports(vault) for c in ingest.read_export(source, path)
               if not (done.get(c["id"], {}).get("source_hash") == c["hash"]
                       and done.get(c["id"], {}).get("prompt_version") == extract.PROMPT_VERSION
                       and done.get(c["id"], {}).get("model") == model))


def lms(*args):
    return subprocess.run([str(LMS), *args], capture_output=True, text=True, timeout=600)


def ensure_model(model):
    """Start the LM Studio server and load the model if needed. Returns what to undo later."""
    started = {"server": False, "model": False}
    state = lms("server", "status")
    if "ON" not in state.stdout + state.stderr:
        lms("server", "start")
        started["server"] = True
    if not any(line.split()[:1] == [model] for line in lms("ps").stdout.splitlines()):
        lms("load", model, "--context-length", "32000", "-y")
        started["model"] = True
    return started


def release_model(model, started):
    if started["model"]:
        lms("unload", model)
    if started["server"]:
        lms("server", "stop")


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--vault", type=Path, default=Path.home() / "personal_chats")
    parser.add_argument("--downloads", type=Path, default=DOWNLOADS)
    parser.add_argument("--model", default=extract.DEFAULT_MODEL)
    parser.add_argument("--no-extract", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    started_at = datetime.now(timezone.utc)

    found = find_exports(args.downloads)
    for provider in status.PROVIDERS:
        if provider in found:
            newest, kind, location = found[provider]
            print(f"{provider}: newest export {location} ({kind}, latest conversation {newest.date()})")
        else:
            print(f"{provider}: no export found in {args.downloads}")
    if args.dry_run:
        return 0

    exports = [(p, materialize(p, kind, loc)) for p, (_, kind, loc) in sorted(found.items())]
    print("\n== import")
    ingest.run(exports, args.vault)

    pending = pending_extraction(args.vault, args.model)
    extraction = "skipped"
    if pending and not args.no_extract:
        print(f"\n== extract ({pending} conversations, local model {args.model})")
        started = ensure_model(args.model)
        try:
            extract.run(args.vault, extract.LocalModel(args.model), args.model)
            extraction = "done"
        finally:
            release_model(args.model, started)
    elif not pending:
        extraction = "nothing pending"

    print("\n== saved memories and instructions")
    claude_path = dict(exports).get("claude")
    import_profile.run(args.vault, claude_memory_file(claude_path) if claude_path else None)

    print("\n== flag")
    flag.run(args.vault / "03-Memories", flag.load_rules())
    print("\n== policy")
    policy.run(args.vault, policy.load_policy(), apply=True)
    print("\n== publish")
    counts = publish.Publisher(args.vault).sync()
    print(json.dumps(counts, sort_keys=True))

    print("\n== status")
    rows = status.provider_status(args.vault)
    policy.write_report(args.vault, "Sync Status.md", status.render(rows))
    for row in rows:
        print(f"{row['source']}: {row.get('conversations', 0)} conversations, newest {row.get('newest_activity', '-')}, "
              f"{row.get('pending_extraction', '-')} pending, {row.get('published', 0)} published")
    RUNS.parent.mkdir(parents=True, exist_ok=True)
    with RUNS.open("a", encoding="utf-8") as log:
        log.write(json.dumps({"started": ingest.iso(started_at), "finished": ingest.iso(datetime.now(timezone.utc)),
                              "exports": {p: str(loc) for p, (_, _, loc) in found.items()},
                              "extraction": extraction, "published": counts.get("live")}) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
