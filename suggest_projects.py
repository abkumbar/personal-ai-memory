#!/usr/bin/env python3
"""Suggest projects for unlabeled conversations from the ones you labeled by hand.

  python3 suggest_projects.py evaluate          accuracy on your own labels (writes nothing)
  python3 suggest_projects.py suggest           preview suggestions in 05-Reports
  python3 suggest_projects.py suggest --apply   also write project_suggested to notes

Each conversation is embedded locally (title plus the user's own messages) and compared
with your labeled conversations. A project is suggested only when the most similar
labeled conversations agree and are similar enough.

Suggestions go in `project_suggested` and `project_suggestion_score`, never in `project`,
so they cannot affect approvals until you copy them over. Embeddings are cached in
state/embeddings.json, keyed by a hash of the embedded text.

Standard library only. Requests go to the local LM Studio server only.
"""

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import sys
import urllib.request

import extract
import ingest
import policy

EMBED_MODEL = "text-embedding-nomic-embed-text-v1.5"
EMBED_URL = "http://127.0.0.1:1234/v1/embeddings"
TEXT_CHARS = 6000
NEIGHBOURS = 5
DEFAULT_THRESHOLD = 0.75
CACHE = ingest.STATE_DIR / "embeddings.json"


def conversation_text(conversation):
    user_text = [item["text"] for item in extract.transcript(conversation) if item["who"] == "user"]
    return ("clustering: " + conversation["title"] + "\n\n" + "\n\n".join(user_text))[:TEXT_CHARS]


def embed(texts, opener):
    payload = json.dumps({"model": EMBED_MODEL, "input": texts}).encode("utf-8")
    request = urllib.request.Request(EMBED_URL, payload, {"Content-Type": "application/json"})
    with opener.open(request, timeout=600) as response:
        data = json.load(response)["data"]
    return [item["embedding"] for item in sorted(data, key=lambda item: item["index"])]


def normalize(vector):
    length = math.sqrt(sum(x * x for x in vector)) or 1.0
    return [x / length for x in vector]


def load_conversations(vault):
    """Return [(note path, note lines, conversation)] for every conversation note."""
    by_key = {f"{c['source']}-{c['id']}": c
              for source, path in extract.latest_exports(vault)
              for c in ingest.read_export(source, path)}
    rows = []
    for path in sorted((vault / "02-Conversations").rglob("*.md")):
        lines, _ = ingest.split_note(path.read_text(encoding="utf-8"))
        if lines and policy.value(lines, "source_id"):
            key = f"{policy.value(lines, 'source')}-{policy.value(lines, 'source_id')}"
            if key in by_key:
                rows.append((path, lines, by_key[key]))
    return rows


def vectors_for(rows):
    cache = json.loads(CACHE.read_text(encoding="utf-8")) if CACHE.exists() else {}
    texts = [conversation_text(c) for _, _, c in rows]
    keys = [hashlib.sha256(t.encode("utf-8")).hexdigest() for t in texts]
    missing = [i for i, key in enumerate(keys) if key not in cache]
    if missing:
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        for start in range(0, len(missing), 16):
            batch = missing[start:start + 16]
            for index, vector in zip(batch, embed([texts[i] for i in batch], opener)):
                cache[keys[index]] = normalize(vector)
        CACHE.parent.mkdir(parents=True, exist_ok=True)
        ingest.write_atomic(CACHE, json.dumps(cache))
        CACHE.chmod(0o600)
    return [cache[key] for key in keys]


def predict(vector, labeled, skip=None):
    """Return (project, score): weighted vote of the nearest labeled conversations."""
    scored = sorted(((sum(a * b for a, b in zip(vector, other)), project)
                     for index, (other, project) in enumerate(labeled) if index != skip), reverse=True)
    votes = defaultdict(float)
    for similarity, project in scored[:NEIGHBOURS]:
        votes[project] += similarity
    if not votes:
        return None, 0.0
    project, weight = max(votes.items(), key=lambda item: item[1])
    agreement = weight / sum(votes.values())
    # Score = how similar the closest supporting example is, scaled by how much neighbours agree.
    best = max(s for s, p in scored[:NEIGHBOURS] if p == project)
    return project, round(best * agreement, 3)


def labeled_set(rows, vectors):
    labeled, names = [], {}
    for (_, lines, _), vector in zip(rows, vectors):
        project = policy.project_of(lines)
        if project:
            key = policy.norm(project)
            names.setdefault(key, Counter())[project] += 1
            labeled.append((vector, key))
    display = {key: counts.most_common(1)[0][0] for key, counts in names.items()}
    return labeled, display


def evaluate(vault):
    rows = load_conversations(vault)
    vectors = vectors_for(rows)
    labeled, _ = labeled_set(rows, vectors)
    sizes = Counter(project for _, project in labeled)
    results = [(predict(vector, labeled, skip=i), truth) for i, (vector, truth) in enumerate(labeled)]
    learnable = [(p, t) for p, t in results if sizes[t] > 1]
    print(f"labeled conversations: {len(labeled)}; with a project seen at least twice: {len(learnable)}")
    print("threshold | suggested | correct when suggested")
    for threshold in (0.0, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85):
        made = [(p, t) for (p, score), t in learnable if score >= threshold]
        right = sum(p == t for p, t in made)
        share = f"{right / len(made):.0%}" if made else "-"
        print(f"{threshold:>9} | {len(made):>4} of {len(learnable)} | {share}")


def suggest(vault, threshold, apply):
    rows = load_conversations(vault)
    vectors = vectors_for(rows)
    labeled, display = labeled_set(rows, vectors)
    out, counts, written = [], Counter(), 0
    for (path, lines, conversation), vector in zip(rows, vectors):
        if policy.project_of(lines):
            continue
        project, score = predict(vector, labeled)
        if project and score >= threshold:
            counts[display[project]] += 1
            out.append((score, display[project], path.stem, conversation["title"]))
            if apply:
                text = path.read_text(encoding="utf-8")
                note_lines, body = ingest.split_note(text)
                new = ingest.join_note(ingest.set_keys(note_lines, {
                    "project_suggested": ingest.scalar(display[project]),
                    "project_suggestion_score": str(score)}), body)
                if new != text:
                    ingest.write_atomic(path, new)
                    written += 1
        else:
            counts["(no suggestion)"] += 1
    out.sort(reverse=True)
    report = ["# Project Suggestions", "",
              f"Threshold {threshold}. Generated by suggest_projects.py; rerunning overwrites this note.",
              "Copy a suggestion into `project` to accept it. Suggestions alone never affect approval.", "",
              "| Suggested project | Conversations |", "|---|---:|"] + [
              f"| {policy.cell(k)} | {v} |" for k, v in counts.most_common()] + [
              "", "| Score | Suggested project | Conversation |", "|---:|---|---|"] + [
              f"| {s} | {policy.cell(p)} | [[{stem}\\|{policy.cell(t)}]] |" for s, p, stem, t in out]
    policy.write_report(vault, "Project Suggestions.md", "\n".join(report) + "\n")
    unlabeled = sum(counts.values())
    print(f"unlabeled conversations: {unlabeled}; suggestions: {len(out)}; no suggestion: {counts['(no suggestion)']}")
    print(f"notes written: {written}" if apply else "preview only; details in 05-Reports/Project Suggestions.md")


def accept(vault, apply):
    """Copy project_suggested into an empty project, then drop the suggestion properties."""
    counts = Counter()
    for path in sorted((vault / "02-Conversations").rglob("*.md")):
        text = path.read_text(encoding="utf-8")
        lines, body = ingest.split_note(text)
        suggestion = policy.value(lines or [], "project_suggested")
        if not suggestion:
            continue
        if policy.project_of(lines):
            counts["skipped (project already set)"] += 1
            continue
        new_lines = policy.remove_keys(lines, ("project", "project_suggested", "project_suggestion_score"))
        new_lines = ingest.set_keys(new_lines, {"project": ingest.scalar(suggestion)})
        counts["accepted"] += 1
        if apply:
            ingest.write_atomic(path, ingest.join_note(new_lines, body))
    for key, count in sorted(counts.items()):
        print(f"{'' if apply else '[preview] '}{key}: {count}")


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("command", choices=["evaluate", "suggest", "accept"])
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--vault", type=Path, default=Path.home() / "personal_chats")
    args = parser.parse_args()
    if args.command == "accept":
        accept(args.vault, args.apply)
    elif args.command == "evaluate":
        evaluate(args.vault)
    else:
        suggest(args.vault, args.threshold, args.apply)
    return 0


if __name__ == "__main__":
    sys.exit(main())
