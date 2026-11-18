import json
from pathlib import Path
import tempfile
import unittest

import publish


def memory(memory_id="mem-1", access="approved", extra="", statement="The user chose Postgres for meal logs.",
           flags="[]", attribution="user_stated", project="Northwind"):
    return (f'---\nid: {memory_id}\ntype: memory\ntitle: "Postgres choice"\ncategory: technical_decision\n'
            f'attribution: {attribution}\nproject: "{project}"\nflags: {flags}\nsource: chatgpt\n'
            f'source_conversation_id: "chatgpt-c1"\nsource_message_ids: ["m1"]\ncreated: 2026-01-02\n'
            f'access: {access}\n{extra}---\n\n{statement}\n\n## Evidence\n\nUser said:\n\n> SECRET QUOTE\n\n'
            f'## Source\n\n[[chatgpt-c1|Chat]]\n')


class PublishTest(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        root = Path(temp.name)
        self.vault = root / "vault"
        (self.vault / "03-Memories").mkdir(parents=True)
        (self.vault / "02-Conversations").mkdir()
        (self.vault / "02-Conversations" / "chat.md").write_text(
            "---\nsource: chatgpt\nsource_id: c1\ntype: conversation\naccess: private\n---\nRAW CONVERSATION\n",
            encoding="utf-8")
        self.out = root / "GrokBotContext"
        self.state = root / "state"
        self.publisher = publish.Publisher(self.vault, self.out, self.state, root / "no-policy.json")

    def write(self, text, name="mem-1"):
        (self.vault / "03-Memories" / f"{name}.md").write_text(text, encoding="utf-8")

    def note(self, name="mem-1"):
        return (self.vault / "03-Memories" / f"{name}.md").read_text(encoding="utf-8")

    def published(self, name="mem-1"):
        path = self.out / "memories" / f"{name}.md"
        return path.read_text(encoding="utf-8") if path.exists() else None

    def audit(self):
        path = self.state / "publish_audit.jsonl"
        return [json.loads(l) for l in path.read_text().splitlines()] if path.exists() else []

    def test_manual_approval_published_without_private_parts(self):
        self.write(memory())
        counts = self.publisher.sync()
        text = self.published()
        self.assertIn("The user chose Postgres", text)
        self.assertIn('source_conversation_id: "chatgpt-c1"', text)
        for leaked in ("SECRET QUOTE", "[[", "flags", "RAW CONVERSATION"):
            self.assertNotIn(leaked, text)
        self.assertEqual(counts["manual approvals fingerprinted"], 1)
        self.assertIn("approved_hash:", self.note())
        self.assertIn('publish_status: "published"', self.note())
        line = json.loads((self.out / "memories.jsonl").read_text().splitlines()[0])
        self.assertEqual((line["id"], line["project"]), ("mem-1", "Northwind"))
        self.assertIn("[[mem-1\\|Postgres choice]]", (self.vault / "GrokBot Context.md").read_text())

    def test_private_and_conversations_never_published(self):
        self.write(memory(access="private"))
        self.publisher.sync()
        self.assertEqual(list((self.out / "memories").glob("*.md")), [])

    def test_second_sync_changes_nothing(self):
        self.write(memory())
        self.publisher.sync()
        note, audit = self.note(), self.audit()
        counts = self.publisher.sync()
        self.assertEqual(self.note(), note)
        self.assertEqual(self.audit(), audit)
        self.assertEqual(counts, {"live": 1})

    def test_edit_after_approval_keeps_approved_version(self):
        self.write(memory())
        self.publisher.sync()
        self.write(self.note().replace("Postgres for meal logs.", "Postgres; also my salary is 200k."))
        counts = self.publisher.sync()
        self.assertEqual(counts["held (edited after approval)"], 1)
        self.assertIn("Postgres for meal logs.", self.published())
        self.assertNotIn("salary", self.published())
        self.assertIn("held (edited after approval)", self.note())
        self.publisher.sync()
        self.assertEqual(sum(e["event"] == "hold" for e in self.audit()), 1)

    def test_deleting_fingerprint_reapproves_new_text(self):
        self.write(memory())
        self.publisher.sync()
        edited = self.note().replace("meal logs.", "meal logs and events.")
        edited = "\n".join(l for l in edited.split("\n") if not l.startswith("approved_hash:"))
        self.write(edited)
        self.publisher.sync()
        self.assertIn("meal logs and events.", self.published())

    def test_revoke_removes_published_copy(self):
        self.write(memory())
        self.publisher.sync()
        self.write(self.note().replace("access: approved", "access: private"))
        counts = self.publisher.sync()
        self.assertIsNone(self.published())
        self.assertEqual(counts["removed (no longer approved)"], 1)
        self.assertNotIn("publish_status", self.note())
        self.assertEqual(self.audit()[-1]["event"], "unpublish")
        self.assertEqual((self.out / "memories.jsonl").read_text(), "")

    def test_deleted_note_unpublished(self):
        self.write(memory())
        self.publisher.sync()
        (self.vault / "03-Memories" / "mem-1.md").unlink()
        self.assertEqual(self.publisher.sync()["removed (note deleted)"], 1)
        self.assertIsNone(self.published())

    def test_credentials_held_even_when_approved(self):
        self.write(memory(flags='["credentials"]'), "mem-1")
        self.write(memory("mem-2", statement="The key is sk-abcdefghijklmnopqrstuvwxyz123."), "mem-2")
        counts = self.publisher.sync()
        self.assertEqual(counts["held (credentials)"], 2)
        self.assertIsNone(self.published("mem-1"))
        self.assertIsNone(self.published("mem-2"))

    def test_policy_approval_uses_its_fingerprint(self):
        import policy
        lines, body = policy.ingest.split_note(memory())
        good = policy.content_hash(lines, body)
        self.write(memory(extra=f'approved_by: "policy:hc@v3"\napproved_hash: {good}\n'))
        self.publisher.sync()
        self.assertIn('approved_by: "policy:hc@v3"', self.published())
        self.write(memory("mem-2", extra='approved_by: "policy:hc@v3"\n'), "mem-2")
        self.assertEqual(self.publisher.sync()["held (policy approval missing fingerprint)"], 1)

    def test_ai_suggestion_labelled(self):
        self.write(memory(attribution="ai_suggested"))
        self.publisher.sync()
        self.assertIn("*AI suggestion.*", self.published())

    def test_dry_run_writes_nothing(self):
        self.write(memory())
        before = self.note()
        self.publisher.sync(dry_run=True)
        self.assertEqual(self.note(), before)
        self.assertFalse(self.out.exists())
        self.assertFalse(self.state.exists())


class AliasNameTest(unittest.TestCase):
    def test_published_project_uses_alias_group_name(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            vault = root / "vault"
            (vault / "03-Memories").mkdir(parents=True)
            (vault / "02-Conversations").mkdir()
            (vault / "03-Memories" / "mem-1.md").write_text(memory(project="sleep-coach"), encoding="utf-8")
            rules = root / "policy.json"
            rules.write_text(json.dumps({"version": 1, "rules": [],
                                         "project_aliases": {"sleep-app": ["sleep-coach"]}}), encoding="utf-8")
            publish.Publisher(vault, root / "out", root / "state", rules).sync()
            line = json.loads((root / "out" / "memories.jsonl").read_text().splitlines()[0])
            self.assertEqual(line["project"], "sleep-app")


if __name__ == "__main__":
    unittest.main()
