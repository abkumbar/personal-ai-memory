import json
from pathlib import Path
import tempfile
import unittest

import import_profile as I
import ingest
import policy

MEMORY = {
    "conversations_memory": "**Work and career**\n\nThe user leads applied AI at a health company.\n\nThe user prefers concise answers.",
    "memory_files": [
        {"path": "/areas/sleep-coaching-app.md", "updated_at": "2026-08-06T00:00:00Z",
         "content": "---\nname: Sleep app\nsources: [x]\n---\n- Building **Sleep Coach** for iOS\n  with on-device data\n- v1 launched in August\n"},
        {"path": "/topics/tools.md", "updated_at": "2026-08-14T00:00:00Z", "content": "---\nname: Tools\n---\n- Uses Obsidian\n"}],
    "account_uuid": "x",
}


class ProfileTest(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.vault = self.root / "vault"
        (self.vault / "03-Memories").mkdir(parents=True)
        self.source = self.root / "memories.json"
        self.source.write_text(json.dumps(MEMORY))
        self.state = self.root / "profile.json"

    def run_import(self):
        return I.run(self.vault, self.source, state_path=self.state)

    def notes(self):
        return {p.name: p.read_text() for p in (self.vault / "03-Memories").glob("*.md")}

    def test_items_become_private_profile_memories(self):
        counts = self.run_import()
        self.assertEqual(counts["created"], 5)
        texts = "\n".join(self.notes().values())
        self.assertIn("Building Sleep Coach for iOS with on-device data", texts)
        self.assertIn("Work and career: The user leads applied AI", texts)
        self.assertIn('project: "sleep-coaching-app"', texts)
        self.assertIn("attribution: provider_memory", texts)
        self.assertNotIn("access: approved", texts)
        self.assertEqual(texts.count("category: profile"), 5)
        self.assertEqual(len(list((self.vault / "01-Raw-Exports" / "claude-memory").iterdir())), 1)
        self.assertTrue((self.vault / "04-Profile" / I.CHATGPT_MEMORIES).exists())

    def test_rerun_is_stable_and_keeps_edits(self):
        self.run_import()
        path = next((self.vault / "03-Memories").glob("*.md"))
        path.write_text(path.read_text().replace("access: private", "access: approved"))
        before = self.notes()
        counts = self.run_import()
        self.assertEqual((counts["created"], counts["existing"]), (0, 5))
        self.assertEqual(self.notes(), before)

    def test_removed_item_marked_not_in_source(self):
        self.run_import()
        changed = json.loads(json.dumps(MEMORY))
        changed["memory_files"][1]["content"] = "---\nname: Tools\n---\n- Uses Logseq\n"
        self.source.write_text(json.dumps(changed))
        counts = self.run_import()
        self.assertEqual((counts["created"], counts["no longer in source"]), (1, 1))
        self.assertEqual(sum("in_source: false" in t and "Uses Obsidian" in t for t in self.notes().values()), 1)

    def test_chatgpt_paste_in(self):
        self.run_import()
        path = self.vault / "04-Profile" / I.CHATGPT_INSTRUCTIONS
        path.write_text(path.read_text() + "- Answer in short bullet points\nI am a health AI leader\n")
        counts = self.run_import()
        self.assertEqual(counts["created"], 2)
        added = [t for t in self.notes().values() if "source: chatgpt" in t]
        self.assertTrue(all("attribution: user_stated" in t for t in added))
        self.assertFalse(any("Paste " in t for t in added))

    def test_policy_accepts_and_publish_labels(self):
        import publish
        self.assertIn("provider_memory", policy.ATTRIBUTIONS)
        record = {k: "" for k in ("id", "title", "category", "project", "created", "source",
                                  "source_conversation_id", "approved_by", "approved_hash", "statement")}
        record.update(attribution="provider_memory", source_message_ids=[], title="t")
        self.assertIn("Recorded by the provider", publish.render_published(record))


if __name__ == "__main__":
    unittest.main()
