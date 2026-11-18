import copy
import json
from pathlib import Path
import tempfile
import unittest

import ingest


def claude_chat():
    return {
        "uuid": "abc-1", "name": "Original title", "created_at": "2026-01-02T03:04:05.123456Z",
        "updated_at": "2026-01-02T04:00:00Z",
        "chat_messages": [
            {"uuid": "m1", "sender": "human", "text": "hello", "created_at": "2026-01-02T03:04:05Z",
             "parent_message_uuid": None, "attachments": [
                 {"file_name": "a.txt", "extracted_content": "attached words"}], "files": [],
             "content": [{"type": "text", "text": "hello"},
                         {"type": "injected_prompt_block", "prompt": "injected words"}]},
            {"uuid": "m2", "sender": "assistant", "text": "hi", "created_at": "2026-01-02T03:05:00Z",
             "parent_message_uuid": "m1", "attachments": [], "files": [],
             "content": [{"type": "thinking", "thinking": "secret reasoning"},
                         {"type": "text", "text": "hi ``` there"}]},
        ],
    }


GROK = {"conversations": [{
    "conversation": {"id": "g1", "title": "Grok chat", "create_time": "2026-02-01T00:00:00Z",
                     "modify_time": "2026-02-01T01:00:00Z"},
    "responses": [{"response": {"_id": "r1", "message": "yo", "sender": "ASSISTANT", "model": "x",
                                "create_time": {"$date": {"$numberLong": "1769904000000"}},
                                "thinking_trace": "grok reasoning"}}],
}]}


class IngestTest(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.vault = self.root / "vault"
        self.folder = self.vault / "02-Conversations"
        self.folder.mkdir(parents=True)
        self.state = self.root / "state"
        self.claude_path = self.root / "conversations.json"
        self.grok_path = self.root / "grok.json"
        self.grok_path.write_text(json.dumps(GROK), encoding="utf-8")
        self.export(claude_chat())

    def export(self, chat):
        self.claude_path.write_text(json.dumps([chat]), encoding="utf-8")

    def run_import(self, **kwargs):
        return ingest.run([("claude", self.claude_path), ("grok", self.grok_path)],
                          self.vault, state_dir=self.state, **kwargs)

    def note(self, source="claude"):
        return next(self.folder.glob(f"{source}-*.md"))

    def legacy_note(self):
        """A note as written by the original importer, then reformatted by Obsidian."""
        conversation = ingest.read_export("claude", self.claude_path)[0]
        body = ingest.render_body(conversation, ingest.LEGACY_RENDER)
        path = self.folder / conversation["filename"]
        path.write_text('---\ndate: 2026-01-02\ntitle: Kept\nsource: claude\nsource_id: abc-1\n'
                        f'message_count: 2\nprivacy: private\ngrok_approved: true\npublish: false\n---\n\n{body}\n',
                        encoding="utf-8")
        return path

    def test_new_notes_are_private_and_complete(self):
        self.run_import()
        text = self.note().read_text(encoding="utf-8")
        for expected in ("access: private", "sensitivity: private", "type: conversation",
                         'id: "claude-abc-1"', "topics: []", "attached words", "secret reasoning",
                         "Injected prompt (not written by the user)", "created_at_utc: \"2026-01-02T03:04:05.123456Z\""):
            self.assertIn(expected, text)
        self.assertIn("grok reasoning", self.note("grok").read_text(encoding="utf-8"))
        self.assertIn("## Assistant", self.note("grok").read_text(encoding="utf-8"))

    def test_second_import_changes_nothing(self):
        self.run_import()
        before = self.note().read_bytes()
        counts = self.run_import()
        self.assertEqual(before, self.note().read_bytes())
        self.assertEqual(counts["claude", "unchanged"], 1)
        self.assertEqual(len(list(self.folder.glob("*.md"))), 2)

    def test_legacy_note_keeps_body_and_approval(self):
        path = self.legacy_note()
        body_before = ingest.split_note(path.read_text(encoding="utf-8"))[1]
        counts = self.run_import()
        lines, body = ingest.split_note(path.read_text(encoding="utf-8"))
        self.assertEqual(body_before, body)
        self.assertEqual(counts["claude", "properties added"], 1)
        self.assertIn("grok_approved: true", lines)
        self.assertIn("title: Kept", lines)
        self.assertIn("access: private", lines)

    def test_user_properties_survive_export_change(self):
        self.run_import()
        path = self.note()
        text = path.read_text(encoding="utf-8").replace("access: private", "access: approved")
        text = text.replace("project:\n", "project: Lighthouse\n").replace('title: "Original title"', "title: Mine")
        path.write_text(text, encoding="utf-8")
        changed = claude_chat()
        changed["updated_at"] = "2026-03-01T00:00:00Z"
        changed["chat_messages"].append({
            "uuid": "m3", "sender": "human", "text": "more", "created_at": "2026-03-01T00:00:00Z",
            "parent_message_uuid": "m2", "attachments": [], "files": [],
            "content": [{"type": "text", "text": "more"}]})
        self.export(changed)
        counts = self.run_import()
        result = path.read_text(encoding="utf-8")
        self.assertEqual(counts["claude", "updated"], 1)
        for expected in ("access: approved", "project: Lighthouse", "title: Mine", "more", "message_count: 3"):
            self.assertIn(expected, result)

    def test_edited_body_is_not_overwritten(self):
        self.run_import()
        path = self.note()
        path.write_text(path.read_text(encoding="utf-8") + "\nMy own note\n", encoding="utf-8")
        edited = path.read_bytes()
        changed = copy.deepcopy(claude_chat())
        changed["chat_messages"][0]["content"][0]["text"] = "rewritten"
        self.export(changed)
        counts = self.run_import()
        self.assertEqual(edited, path.read_bytes())
        self.assertTrue(any(s.startswith("conflict") for _, s in counts))
        self.assertEqual(len(list((self.state / "conflicts").glob("*.md"))), 1)

    def test_renamed_note_is_not_duplicated(self):
        self.run_import()
        path = self.note()
        (self.folder / "sub").mkdir()
        path.rename(self.folder / "sub" / "My name.md")
        self.run_import()
        self.assertEqual(len(list(self.folder.rglob("claude-*.md"))), 0)
        self.assertEqual(len(list(self.folder.rglob("*.md"))), 2)

    def test_deleted_note_is_not_recreated(self):
        self.run_import()
        self.note().unlink()
        self.run_import()
        self.assertEqual(list(self.folder.glob("claude-*.md")), [])

    def test_upgrade_only_touches_unedited_legacy_notes(self):
        path = self.legacy_note()
        self.run_import()
        self.run_import(upgrade_bodies=True)
        text = path.read_text(encoding="utf-8")
        self.assertIn("secret reasoning", text)
        self.assertIn("grok_approved: true", text)

    def test_raw_export_archived_once_read_only(self):
        self.run_import()
        self.run_import()
        archived = list((self.vault / "01-Raw-Exports" / "claude").iterdir())
        self.assertEqual(len(archived), 1)
        self.assertEqual(archived[0].stat().st_mode & 0o777, 0o400)

    def test_dry_run_writes_nothing(self):
        self.run_import(dry_run=True)
        self.assertEqual(list(self.folder.iterdir()), [])
        self.assertFalse(self.state.exists())

    def test_bad_export_writes_nothing(self):
        self.grok_path.write_text('{"conversations": [{}]}', encoding="utf-8")
        with self.assertRaises(KeyError):
            self.run_import()
        self.assertEqual(list(self.folder.iterdir()), [])


if __name__ == "__main__":
    unittest.main()
