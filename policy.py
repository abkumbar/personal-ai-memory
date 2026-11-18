#!/usr/bin/env python3
"""Approve memory notes automatically from a written policy.

  python3 policy.py projects        list project names to write rules and aliases against
  python3 policy.py run             preview decisions (writes only report notes)
  python3 policy.py run --apply     apply them

Each memory gets one outcome: approve, private, or review.
- private beats review, and review beats approve.
- A flag blocks approval unless the rule lists it in allow_flags. Flags in
  never_auto_approve_flags can never be allowed.
- Only attributions in auto_approve_attribution are auto-approved; others go to review.
- Keyword flags are refreshed before deciding.
- Notes the policy manages get `policy_outcome` (approve/private/review) and `policy_reason`, so
  the Pending Review view can show only what needs a decision. Notes it does not manage
  (manual approvals, rejected, revoked by you) are left untouched.
- A memory with no project of its own takes its conversation's project at decision time.
  Nothing is copied, so changing a conversation's project changes the outcome on the next run.

What it changes:
- `access: private` (undecided) to `approved`, recording approved_by, approved_hash and approved_at.
- Policy approvals the policy no longer supports go back to `private`.
- If an approved note was edited and the policy still approves it, the approval is renewed
  for the new content. Either way the change goes in the audit log.
Never touched: manual approvals, any other access value such as `rejected`, and
policy approvals you revoked yourself (access changed, approved_by kept).

Standard library only.
"""

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import random
import re
import sys

import extract
import flag
import ingest

POLICY_PATH = Path(__file__).resolve().parent / "approval_policy.json"
# Extraction origins plus memories recorded by a provider (Claude/ChatGPT saved memory).
ATTRIBUTIONS = list(extract.ORIGINS) + ["provider_memory"]
REPORTS = "05-Reports"
ACTIONS = ("approve", "private", "review")
RANK = {"approve": 0, "review": 1, "private": 2}
RULE_KEYS = {"id", "action", "projects", "categories", "flags", "subjects", "attributions", "allow_flags"}
OUTCOME_KEYS = ("policy_outcome", "policy_reason")
POLICY_KEYS = {"version", "never_auto_approve_flags", "auto_approve_attribution", "default",
               "project_aliases", "rules"}
APPROVAL_KEYS = ("approved_by", "approved_hash", "approved_at")


def norm(text):
    return re.sub(r"[^a-z0-9]+", "", str(text).casefold())


# --- Policy -----------------------------------------------------------------

def load_policy(path=POLICY_PATH):
    policy = {k: v for k, v in json.loads(path.read_text(encoding="utf-8")).items()
              if not k.startswith("_")}
    errors = [f"unknown policy key: {k}" for k in policy if k not in POLICY_KEYS]
    never = set(policy.get("never_auto_approve_flags", []))
    if not isinstance(policy.get("version"), int):
        errors.append("version must be an integer")
    if policy.get("default", "review") not in ACTIONS:
        errors.append("default must be approve, private or review")
    for name in never | {f for r in policy.get("rules", []) for f in r.get("flags", []) + r.get("allow_flags", [])}:
        if name not in extract.FLAGS:
            errors.append(f"unknown flag: {name}")
    for name in policy.get("auto_approve_attribution", []) + [
            a for r in policy.get("rules", []) for a in r.get("attributions", [])]:
        if name not in ATTRIBUTIONS:
            errors.append(f"unknown attribution: {name}")
    ids = Counter(r.get("id") for r in policy.get("rules", []))
    for rule in policy.get("rules", []):
        label = rule.get("id", "?")
        errors += [f"rule {label}: unknown key {k}" for k in rule
                   if k not in RULE_KEYS and not k.startswith("_")]
        if ids[rule.get("id")] > 1 or not rule.get("id"):
            errors.append(f"rule {label}: id missing or duplicated")
        if rule.get("action") not in ACTIONS:
            errors.append(f"rule {label}: action must be approve, private or review")
        errors += [f"rule {label}: unknown category {c}" for c in rule.get("categories", [])
                   if c not in extract.CATEGORIES]
        if set(rule.get("allow_flags", [])) & never:
            errors.append(f"rule {label}: allow_flags includes a never_auto_approve flag")
        if rule.get("allow_flags") and rule.get("action") != "approve":
            errors.append(f"rule {label}: allow_flags only applies to approve rules")
    # Aliases may chain (a -> b -> c); each name resolves to the end of its chain.
    direct = {}
    for canonical, others in policy.get("project_aliases", {}).items():
        for name in others:
            key = norm(name)
            if key == norm(canonical):
                continue
            if key in direct and direct[key] != norm(canonical):
                errors.append(f"alias {name!r} is listed under two projects")
            direct[key] = norm(canonical)
    aliases = {}
    for key in direct:
        seen, target = {key}, direct[key]
        while target in direct:
            if target in seen:
                errors.append(f"aliases form a loop through {target!r}")
                break
            seen.add(target)
            target = direct[target]
        aliases[key] = target
    if errors:
        raise SystemExit("Policy errors:\n  " + "\n  ".join(errors))
    policy["_aliases"] = aliases
    policy["_names"] = {norm(c): c for c in policy.get("project_aliases", {})}
    return policy


def canonical_project(policy, project):
    key = norm(project)
    return policy["_aliases"].get(key, key)


def display_project(policy, project):
    """The project name to show: the alias group's name if it has one, else as written."""
    return policy["_names"].get(canonical_project(policy, project), project) if project else project


def rule_matches(policy, rule, memory):
    if "projects" in rule and canonical_project(policy, memory["project"]) not in {
            canonical_project(policy, p) for p in rule["projects"]}:
        return False
    if "categories" in rule and memory["category"] not in rule["categories"]:
        return False
    if "flags" in rule and not set(rule["flags"]) & memory["flags"]:
        return False
    if "subjects" in rule and not any(memory["subject"].startswith(s) for s in rule["subjects"]):
        return False
    if "attributions" in rule and memory["attribution"] not in rule["attributions"]:
        return False
    return True


def decide(policy, memory):
    """Return (outcome, rule id or reason)."""
    matched = [r for r in policy["rules"] if rule_matches(policy, r, memory)]
    candidates = []
    for rule in matched:
        if rule["action"] != "approve":
            candidates.append((rule["action"], rule["id"]))
            continue
        blocking = memory["flags"] - set(rule.get("allow_flags", []))
        if memory["flags"] & set(policy.get("never_auto_approve_flags", [])):
            candidates.append(("private", f"{rule['id']} (never-approve flag)"))
        elif blocking:
            candidates.append(("review", f"{rule['id']} (flag: {', '.join(sorted(blocking))})"))
        elif memory["attribution"] not in policy.get("auto_approve_attribution", []):
            candidates.append(("review", f"{rule['id']} (attribution: {memory['attribution']})"))
        else:
            candidates.append(("approve", rule["id"]))
    if not candidates:
        return policy.get("default", "review"), "default"
    return max(candidates, key=lambda c: RANK[c[0]])


# --- Notes ------------------------------------------------------------------

def value(lines, key):
    for line in lines:
        match = re.match(rf"{re.escape(key)}:\s?(.*)$", line)
        if match:
            raw = match.group(1).strip()
            if raw.startswith('"'):
                try:
                    return json.loads(raw)
                except ValueError:
                    return raw.strip('"')
            return raw.strip("'")
    return ""


def remove_keys(lines, keys):
    return [line for line in lines if not any(re.match(rf"{re.escape(k)}:(\s|$)", line) for k in keys)]


def content_hash(lines, body):
    return hashlib.sha256((value(lines, "title") + "\n" + body.strip()).encode("utf-8")).hexdigest()


def project_of(lines):
    """A note's project, whether written as text, a flow list or an Obsidian block list."""
    raw = value(lines, "project")
    if raw and not raw.startswith("["):
        return raw
    items = flag.read_list(lines, "project")[0]
    return next((item for item in items if item), "")


def conversation_projects(vault):
    """Map "<source>-<source_id>" to the project set on each conversation note."""
    found = {}
    for path in (vault / "02-Conversations").rglob("*.md"):
        lines, _ = ingest.split_note(path.read_text(encoding="utf-8"))
        if lines and value(lines, "source_id"):
            project = project_of(lines)
            if project:
                found[f"{value(lines, 'source')}-{value(lines, 'source_id')}"] = project
    return found


def read_memory(lines, body, conv_projects=None):
    own = project_of(lines)
    inherited = (conv_projects or {}).get(value(lines, "source_conversation_id"), "")
    return {
        "id": value(lines, "id"), "title": value(lines, "title"),
        "project": own or inherited,
        "project_from": "memory" if own else ("conversation" if inherited else ""),
        "category": value(lines, "category"),
        "attribution": value(lines, "attribution"), "subject": value(lines, "subject"),
        "flags": set(flag.read_list(lines, "flags")[0]), "access": value(lines, "access"),
        "approved_by": value(lines, "approved_by"), "approved_hash": value(lines, "approved_hash"),
        "hash": content_hash(lines, body),
    }


def plan_note(text, policy, keyword_rules, conv_projects=None):
    """Return (event, outcome, reason, memory, new text). event is None when nothing changes."""
    text, _ = flag.flag_note(text, keyword_rules)
    lines, body = ingest.split_note(text)
    memory = read_memory(lines, body, conv_projects)
    outcome, reason = decide(policy, memory)
    by_policy = memory["approved_by"].startswith("policy:")
    stamp = f"policy:{reason}@v{policy['version']}"
    approval = {"access": "approved", "approved_by": ingest.scalar(stamp),
                "approved_hash": memory["hash"],
                "approved_at": ingest.scalar(ingest.iso(datetime.now(timezone.utc)))}

    if memory["access"] == "approved" and not by_policy:
        return "manual (untouched)", outcome, reason, memory, text
    if memory["access"] != "approved" and by_policy:
        return "revoked by you (untouched)", outcome, reason, memory, text
    if memory["access"] not in ("private", "approved"):
        return f"access {memory['access']} (untouched)", outcome, reason, memory, text

    # From here on the policy manages this note, so record its decision on it.
    tags = {"policy_outcome": outcome, "policy_reason": ingest.scalar(reason)}
    tagged = ingest.set_keys(lines, tags)

    if memory["access"] == "private":
        if outcome != "approve":
            return None, outcome, reason, memory, ingest.join_note(tagged, body)
        return "approve", outcome, reason, memory, ingest.join_note(ingest.set_keys(tagged, approval), body)

    # Currently approved by policy.
    if outcome != "approve":
        lines = ingest.set_keys(remove_keys(tagged, APPROVAL_KEYS), {"access": "private"})
        return "revoke", outcome, reason, memory, ingest.join_note(lines, body)
    if memory["approved_hash"] != memory["hash"] or value(lines, "approved_by") != stamp:
        event = "reapprove (edited)" if memory["approved_hash"] != memory["hash"] else "reapprove (policy changed)"
        return event, outcome, reason, memory, ingest.join_note(ingest.set_keys(tagged, approval), body)
    return None, outcome, reason, memory, ingest.join_note(tagged, body)


# --- Commands ---------------------------------------------------------------

def memory_notes(vault):
    for path in sorted((vault / "03-Memories").glob("*.md")):
        text = path.read_text(encoding="utf-8")
        lines, _ = ingest.split_note(text)
        if lines and "type: memory" in lines:
            yield path, text


def write_report(vault, name, content):
    folder = vault / REPORTS
    folder.mkdir(exist_ok=True)
    ingest.write_atomic(folder / name, content)


def cell(text):
    return str(text).replace("|", "\\|").replace("\n", " ")


def projects(vault, policy):
    groups = defaultdict(Counter)
    conv_projects = conversation_projects(vault)
    inherited = 0
    for _, text in memory_notes(vault):
        lines, body = ingest.split_note(text)
        memory = read_memory(lines, body, conv_projects)
        project = memory["project"]
        inherited += memory["project_from"] == "conversation"
        groups[canonical_project(policy, project) if project else ""][project or "(none)"] += 1
    rows = sorted(groups.items(), key=lambda g: -sum(g[1].values()))
    out = ["# Projects", "", "Generated by policy.py; rerunning overwrites this note.",
           "Names on one row are treated as the same project. To merge rows, add aliases in "
           "approval_policy.json under project_aliases.",
           f"Counts include {inherited} memories with no project of their own that take the "
           f"project of their conversation ({len(conv_projects)} conversations have one set).", "",
           "| Memories | Project names as written |", "|---:|---|"]
    for _, names in rows:
        out.append(f"| {sum(names.values())} | " + "; ".join(
            f"{cell(n)} ({c})" for n, c in names.most_common()) + " |")
    write_report(vault, "Projects.md", "\n".join(out) + "\n")
    print(f"projects: {len(rows)} distinct (after aliases), "
          f"{sum(1 for k in groups if k)} named; written to {REPORTS}/Projects.md")


def run(vault, policy, apply=False, state_dir=ingest.STATE_DIR, sample_size=20):
    keyword_rules = flag.load_rules()
    conv_projects = conversation_projects(vault)
    events, outcomes, by_rule, project_from = Counter(), Counter(), Counter(), Counter()
    approved, audit = [], []
    for path, text in memory_notes(vault):
        event, outcome, reason, memory, new_text = plan_note(text, policy, keyword_rules, conv_projects)
        outcomes[outcome] += 1
        project_from[memory["project_from"] or "none"] += 1
        by_rule[f"{outcome}: {reason}"] += 1
        if event:
            events[event] += 1
        if event == "approve":
            approved.append((memory, reason))
        if event in ("approve", "revoke") or (event or "").startswith("reapprove"):
            audit.append({"at": ingest.iso(datetime.now(timezone.utc)), "event": event,
                          "memory": memory["id"], "rule": reason,
                          "policy_version": policy["version"], "content_hash": memory["hash"]})
        if new_text != text:
            events["notes to update"] += 1
            if apply:
                ingest.write_atomic(path, new_text)

    if apply and audit:
        state_dir.mkdir(parents=True, exist_ok=True)
        with (state_dir / "approval_audit.jsonl").open("a", encoding="utf-8") as log:
            for entry in audit:
                log.write(json.dumps(entry) + "\n")

    sample = random.Random(policy["version"]).sample(approved, min(sample_size, len(approved)))
    mode = "Applied" if apply else "Preview (nothing applied)"
    out = [f"# Policy Preview", "", f"{mode}, policy version {policy['version']}. "
           "Generated by policy.py; rerunning overwrites this note.", "",
           "## Changes", ""] + [f"- {k}: {v}" for k, v in sorted(events.items())] + [
           "", "## Where each memory's project came from", ""] + [
           f"- {k}: {v}" for k, v in sorted(project_from.items())] + [
           "", "## Outcome by rule", "", "| Memories | Outcome: rule |", "|---:|---|"] + [
           f"| {v} | {cell(k)} |" for k, v in by_rule.most_common()] + [
           "", f"## Spot check: {len(sample)} random new approvals", "",
           "| Memory | Rule | Project | Category | Attribution |", "|---|---|---|---|---|"] + [
           f"| [[{m['id']}\\|{cell(m['title'])}]] | {cell(r)} | {cell(m['project'])}"
           f"{' (from conversation)' if m['project_from'] == 'conversation' else ''} | {m['category']} | {m['attribution']} |"
           for m, r in sample]
    write_report(vault, "Policy Preview.md", "\n".join(out) + "\n")

    prefix = "" if apply else "[preview] "
    for key, count in sorted(events.items()):
        print(f"{prefix}{key}: {count}")
    for key, count in sorted(outcomes.items()):
        print(f"{prefix}outcome {key}: {count}")
    for key, count in sorted(project_from.items()):
        print(f"{prefix}project from {key}: {count}")
    print(f"{prefix}details in {REPORTS}/Policy Preview.md")
    return events, outcomes


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("command", choices=["projects", "run"])
    parser.add_argument("--apply", action="store_true", help="write approvals (run only)")
    parser.add_argument("--vault", type=Path, default=Path.home() / "personal_chats")
    parser.add_argument("--policy", type=Path, default=POLICY_PATH)
    args = parser.parse_args()
    policy = load_policy(args.policy)
    if args.command == "projects":
        projects(args.vault, policy)
    else:
        run(args.vault, policy, apply=args.apply)
    return 0


if __name__ == "__main__":
    sys.exit(main())
