import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock
import zipfile

import ingest
import status
import sync

CLAUDE = [{"uuid": "c1", "name": "Claude chat", "created_at": "2026-10-01T00:00:00Z", "updated_at": "2026-10-02T00:00:00Z",
           "chat_messages": [{"uuid": "m1", "sender": "human", "text": "hi", "created_at": "2026-10-01T00:00:00Z",
                              "attachments": [], "files": [], "content": [{"type": "text", "text": "hi"}]}]}]
GPT = [{"conversation_id": "g1", "id": "g1", "title": "GPT chat", "create_time": 1767225600.0,
        "update_time": 1767229200.0, "current_node": "u", "mapping": {
            "r": {"id": "r", "parent": None, "children": ["u"], "message": None},
            "u": {"id": "u", "parent": "r", "children": [], "message": {
                "id": "u", "author": {"role": "user"}, "create_time": 1767225600.0,
                "content": {"content_type": "text", "parts": ["hello"]}, "metadata": {}}}}}]
GROK = {"conversations": [{"conversation": {"id": "k1", "title": "Grok chat", "create_time": "2026-03-01T00:00:00Z",
                                            "modify_time": "2026-03-01T00:00:00Z"},
                           "responses": [{"response": {"_id": "r1", "message": "yo", "sender": "human",
                                                       "create_time": {"$date": {"$numberLong": "1772323200000"}}}}]}]}


class SyncTest(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.downloads = self.root / "Downloads"
        (self.downloads / "gpt").mkdir(parents=True)
        (self.downloads / "gpt" / "conversations-000.json").write_text(json.dumps(GPT))
        (self.downloads / "gpt" / "user.json").write_text('{"email": "x"}')
        (self.downloads / "deep" / "grok").mkdir(parents=True)
        (self.downloads / "deep" / "grok" / "prod-grok-backend.json").write_text(json.dumps(GROK))
        with zipfile.ZipFile(self.downloads / "data-claude.zip", "w") as archive:
            archive.writestr("conversations.json", json.dumps(CLAUDE))
            archive.writestr("memories/abc.json", json.dumps({"conversations_memory": "x", "memory_files": []}))
            archive.writestr("users.json", "[]")
            archive.writestr("../../escape/conversations.json", json.dumps(CLAUDE))
        (self.downloads / "notes.json").write_text('{"mapping": 1}')

    def test_finds_each_provider_by_content(self):
        found = sync.find_exports(self.downloads)
        self.assertEqual(set(found), {"claude", "chatgpt", "grok"})
        self.assertEqual(found["claude"][1], "zip")
        self.assertEqual(found["chatgpt"][1:], ("dir", self.downloads / "gpt"))
        self.assertEqual(found["grok"][2].name, "prod-grok-backend.json")

    def test_zip_unpacked_safely_and_importable(self):
        with mock.patch.object(sync, "INBOX", self.root / "state" / "inbox"):
            path = sync.materialize("claude", "zip", self.downloads / "data-claude.zip")
            self.assertEqual(path.name, "conversations.json")
            self.assertEqual(sorted(p.name for p in path.parent.iterdir()), ["abc.json", "conversations.json"])
            self.assertEqual(sync.claude_memory_file(path).name, "abc.json")
            self.assertFalse((self.root / "escape").exists())
            self.assertEqual(len(ingest.read_export("claude", path)), 1)
            self.assertEqual(sync.materialize("claude", "zip", self.downloads / "data-claude.zip"), path)

    def test_status_reports_freshness_gaps_and_pending(self):
        vault, state = self.root / "vault", self.root / "state"
        (vault / "02-Conversations").mkdir(parents=True)
        (vault / "03-Memories").mkdir()
        with mock.patch.object(ingest, "STATE_DIR", state):
            ingest.run([("claude", self.downloads / "gpt" / ".." / "gpt" / "conversations-000.json")][:0] +
                       [("grok", self.downloads / "deep" / "grok" / "prod-grok-backend.json")], vault, state_dir=state)
            from datetime import datetime, timezone
            rows = {r["source"]: r for r in status.provider_status(vault, now=datetime(2026, 11, 14, tzinfo=timezone.utc))}
        self.assertEqual(rows["grok"]["conversations"], 1)
        self.assertEqual(rows["grok"]["pending_extraction"], 1)
        self.assertTrue(any("No activity" in w for w in rows["grok"]["warnings"]))
        self.assertEqual(rows["claude"]["warnings"], ["No export imported yet."])
        report = status.render(list(rows.values()))
        self.assertIn("| grok | 1 |", report)


class NewestTest(unittest.TestCase):
    def test_picks_export_with_newest_conversation_not_newest_file(self):
        import os
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            older, newer = root / "a", root / "b"
            older.mkdir(); newer.mkdir()
            fresh = json.loads(json.dumps(CLAUDE)); fresh[0]["updated_at"] = "2026-11-10T00:00:00Z"
            (older / "conversations.json").write_text(json.dumps(CLAUDE))
            (newer / "conversations.json").write_text(json.dumps(fresh))
            os.utime(newer / "conversations.json", (315532800, 315532800))  # 1980, like zip extracts
            found = sync.find_exports(root)
            self.assertEqual(found["claude"][2], newer / "conversations.json")


if __name__ == "__main__":
    unittest.main()
