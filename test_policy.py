import json
from pathlib import Path
import tempfile
import unittest

import flag
import ingest
import policy as P

POLICY = {
    "version": 1,
    "never_auto_approve_flags": ["credentials", "family", "location", "legal"],
    "auto_approve_attribution": ["user_stated", "user_confirmed_ai_suggestion"],
    "default": "review",
    "project_aliases": {"Northwind": ["Health Canvas", "health-canvas app"]},
    "rules": [
        {"id": "hc", "action": "approve", "projects": ["Northwind"],
         "categories": ["technical_decision", "project_context"], "allow_flags": ["health"]},
        {"id": "career", "action": "private", "categories": ["career"]},
        {"id": "employer", "action": "private", "flags": ["employer"]},
    ],
}
KEYWORDS = flag.load_rules()


def note(project="Health Canvas", category="technical_decision", attribution="user_stated",
         flags="[]", access="private", statement="The user chose Postgres for meal logs.", extra=""):
    return (f'---\nid: mem-1\ntype: memory\ntitle: "Postgres choice"\ncategory: {category}\n'
            f'attribution: {attribution}\nsubject: "db"\nproject: "{project}"\nflags: {flags}\n'
            f"access: {access}\n{extra}---\n\n{statement}\n\n## Source\n\n[[c|Chat]]\n")


def load(policy=POLICY):
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as file:
        json.dump(policy, file)
    return P.load_policy(Path(file.name))


class DecideTest(unittest.TestCase):
    def plan(self, text, policy=None):
        return P.plan_note(text, policy or load(), KEYWORDS)

    def test_alias_match_approves_with_stamp_and_hash(self):
        event, outcome, reason, memory, text = self.plan(note())
        self.assertEqual((event, outcome, reason), ("approve", "approve", "hc"))
        self.assertIn("access: approved", text)
        self.assertIn('approved_by: "policy:hc@v1"', text)
        self.assertIn(f"approved_hash: {memory['hash']}", text)

    def test_allowed_flag_does_not_block(self):
        self.assertEqual(self.plan(note(flags='["health"]'))[1], "approve")

    def test_other_flag_sends_to_review(self):
        self.assertEqual(self.plan(note(flags='["financial"]'))[1:3], ("review", "hc (flag: financial)"))

    def test_keyword_flags_are_refreshed_before_deciding(self):
        outcome = self.plan(note(statement="The user's salary is $150k."))[1]
        self.assertEqual(outcome, "review")

    def test_never_flag_stays_private(self):
        self.assertEqual(self.plan(note(flags='["family"]'))[1], "private")

    def test_ai_suggestion_goes_to_review(self):
        self.assertEqual(self.plan(note(attribution="ai_suggested"))[1], "review")

    def test_private_rule_beats_approve(self):
        self.assertEqual(self.plan(note(flags='["employer"]'))[1:3], ("private", "employer"))

    def test_unmatched_uses_default(self):
        self.assertEqual(self.plan(note(project="Other"))[1:3], ("review", "default"))

    def test_rejected_and_manual_approvals_untouched(self):
        rejected = note(access="rejected")
        self.assertEqual(self.plan(rejected)[0], "access rejected (untouched)")
        self.assertEqual(self.plan(rejected)[4], rejected)
        manual = note(project="Other", access="approved")
        self.assertEqual(self.plan(manual)[0], "manual (untouched)")
        self.assertEqual(self.plan(manual)[4], manual)

    def test_user_revocation_is_sticky(self):
        approved = self.plan(note())[4]
        revoked = approved.replace("access: approved", "access: private")
        event, _, _, _, text = self.plan(revoked)
        self.assertEqual(event, "revoked by you (untouched)")
        self.assertEqual(text, revoked)

    def test_policy_change_revokes_and_clears_stamp(self):
        approved = self.plan(note())[4]
        stricter = dict(POLICY, version=2, rules=POLICY["rules"][1:])
        event, outcome, _, _, text = self.plan(approved, load(stricter))
        self.assertEqual((event, outcome), ("revoke", "review"))
        self.assertIn("access: private", text)
        self.assertNotIn("approved_by", text)

    def test_edit_after_approval_renews_or_revokes(self):
        approved = self.plan(note())[4]
        self.assertIsNone(self.plan(approved)[0])
        edited = approved.replace("Postgres for meal logs.", "Postgres for meal logs and events.")
        self.assertEqual(self.plan(edited)[0], "reapprove (edited)")
        leaked = approved.replace("Postgres for meal logs.", "Postgres; my wife agreed.")
        event, outcome, _, _, text = self.plan(leaked)
        self.assertEqual((event, outcome), ("revoke", "private"))
        self.assertIn("access: private", text)

    def test_version_bump_restamps(self):
        approved = self.plan(note())[4]
        event, _, _, _, text = self.plan(approved, load(dict(POLICY, version=2)))
        self.assertEqual(event, "reapprove (policy changed)")
        self.assertIn("policy:hc@v2", text)

    def test_invalid_policies_rejected(self):
        bad = [
            dict(POLICY, rules=[{"id": "x", "action": "approve", "allow_flags": ["family"]}]),
            dict(POLICY, rules=[{"id": "x", "action": "maybe"}]),
            dict(POLICY, rules=[{"id": "x", "action": "approve", "categories": ["nope"]}]),
            dict(POLICY, rules=[{"id": "x", "action": "approve", "project": ["typo key"]}]),
            dict(POLICY, never_auto_approve_flags=["weather"]),
        ]
        for policy in bad:
            with self.assertRaises(SystemExit):
                load(policy)


class RunTest(unittest.TestCase):
    def test_preview_writes_only_reports_and_apply_audits(self):
        with tempfile.TemporaryDirectory() as temp:
            vault, state = Path(temp) / "vault", Path(temp) / "state"
            (vault / "03-Memories").mkdir(parents=True)
            path = vault / "03-Memories" / "mem-1.md"
            path.write_text(note(), encoding="utf-8")
            (vault / "03-Memories" / "Memories.base").write_text("views: []\n", encoding="utf-8")
            P.run(vault, load(), apply=False, state_dir=state)
            self.assertEqual(path.read_text(encoding="utf-8"), note())
            self.assertFalse(state.exists())
            self.assertTrue((vault / "05-Reports" / "Policy Preview.md").exists())
            P.run(vault, load(), apply=True, state_dir=state)
            self.assertIn("access: approved", path.read_text(encoding="utf-8"))
            entry = json.loads((state / "approval_audit.jsonl").read_text(encoding="utf-8"))
            self.assertEqual((entry["event"], entry["memory"], entry["rule"]), ("approve", "mem-1", "hc"))
            P.projects(vault, load())
            self.assertIn("Health Canvas (1)", (vault / "05-Reports" / "Projects.md").read_text(encoding="utf-8"))


class InheritTest(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.vault = Path(temp.name) / "vault"
        self.state = Path(temp.name) / "state"
        (self.vault / "03-Memories").mkdir(parents=True)
        (self.vault / "02-Conversations").mkdir()
        self.conversation = self.vault / "02-Conversations" / "renamed by user.md"
        self.memory = self.vault / "03-Memories" / "mem-1.md"

    def conv(self, project_lines):
        self.conversation.write_text(
            f'---\nsource: claude\nsource_id: "abc"\n{project_lines}\naccess: private\n---\nbody\n',
            encoding="utf-8")

    def mem(self, project='project:'):
        text = note().replace('project: "Health Canvas"', project).replace(
            "access: private", 'source_conversation_id: "claude-abc"\naccess: private')
        self.memory.write_text(text, encoding="utf-8")

    def run_policy(self):
        P.run(self.vault, load(), apply=True, state_dir=self.state)
        return self.memory.read_text(encoding="utf-8")

    def test_blank_memory_inherits_conversation_project(self):
        self.conv("project: Northwind")
        self.mem()
        text = self.run_policy()
        self.assertIn("access: approved", text)
        self.assertIn("project:\n", text)  # Inherited, not copied.

    def test_obsidian_list_project_on_conversation(self):
        self.conv("project:\n  - Health Canvas")
        self.mem()
        self.assertIn("access: approved", self.run_policy())

    def test_memory_project_wins(self):
        self.conv("project: Northwind")
        self.mem('project: "Something Else"')
        self.assertIn("access: private", self.run_policy())

    def test_changing_conversation_project_revokes(self):
        self.conv("project: Northwind")
        self.mem()
        self.run_policy()
        self.conv("project: Unrelated")
        text = self.run_policy()
        self.assertIn("access: private", text)
        self.assertNotIn("approved_by", text)

    def test_no_project_anywhere_uses_default(self):
        self.conv("project:")
        self.mem()
        self.assertIn("access: private", self.run_policy())

    def test_projects_report_counts_inherited(self):
        self.conv("project: Northwind")
        self.mem()
        P.projects(self.vault, load())
        report = (self.vault / "05-Reports" / "Projects.md").read_text(encoding="utf-8")
        self.assertIn("1 memories with no project of their own", report)
        self.assertIn("Northwind (1)", report)


class AliasTest(unittest.TestCase):
    def test_chained_aliases_resolve_in_any_order(self):
        for aliases in ({"cgm-transformer": ["cgm-transformer-fda"], "cgm": ["cgm-transformer"]},
                        {"cgm": ["cgm-transformer"], "cgm-transformer": ["cgm-transformer-fda"]}):
            policy = load(dict(POLICY, project_aliases=aliases))
            self.assertEqual(P.canonical_project(policy, "CGM Transformer FDA"), "cgm")
            self.assertEqual(P.canonical_project(policy, "cgm-transformer"), "cgm")
            self.assertEqual(P.canonical_project(policy, "cgm"), "cgm")

    def test_alias_conflicts_and_loops_rejected(self):
        for aliases in ({"a": ["x"], "b": ["x"]}, {"a": ["b"], "b": ["a"]}):
            with self.assertRaises(SystemExit):
                load(dict(POLICY, project_aliases=aliases))


class OutcomeTagTest(unittest.TestCase):
    def plan(self, text, policy=None):
        return P.plan_note(text, policy or load(), KEYWORDS)

    def test_ai_suggestion_rule_beats_approve(self):
        policy = load(dict(POLICY, auto_approve_attribution=["user_stated", "ai_suggested"],
                           rules=POLICY["rules"] + [{"id": "ai", "action": "private", "attributions": ["ai_suggested"]}]))
        event, outcome, reason, _, text = self.plan(note(attribution="ai_suggested"), policy)
        self.assertEqual((event, outcome, reason), (None, "private", "ai"))
        self.assertIn("policy_outcome: private", text)
        self.assertEqual(self.plan(note(), policy)[1], "approve")

    def test_unknown_attribution_rejected(self):
        with self.assertRaises(SystemExit):
            load(dict(POLICY, rules=[{"id": "x", "action": "private", "attributions": ["robot"]}]))

    def test_tags_written_and_stable(self):
        _, _, _, _, text = self.plan(note(project="Other"))
        self.assertIn("policy_outcome: review", text)
        self.assertIn('policy_reason: "default"', text)
        self.assertEqual(self.plan(text)[4], text)
        approved = self.plan(note())[4]
        self.assertIn("policy_outcome: approve", approved)
        self.assertEqual(self.plan(approved)[4], approved)

    def test_tags_follow_revocation(self):
        approved = self.plan(note())[4]
        revoked = self.plan(approved, load(dict(POLICY, version=2, rules=POLICY["rules"][1:])))[4]
        self.assertIn("policy_outcome: review", revoked)
        self.assertNotIn("policy_outcome: approve", revoked)

    def test_unmanaged_notes_get_no_tags(self):
        for text in (note(access="rejected"), note(project="Other", access="approved")):
            self.assertNotIn("policy_outcome", self.plan(text)[4])


if __name__ == "__main__":
    unittest.main()
