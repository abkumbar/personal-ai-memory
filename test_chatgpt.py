import json
from pathlib import Path
import tempfile
import unittest

import extract
import ingest


def node(node_id, parent, role=None, content=None, created=1767225600.0, children=()):
    message = None
    if role:
        message = {"id": node_id, "author": {"role": role, "name": None}, "create_time": created,
                   "content": content, "metadata": {"model_slug": "gpt-x"}}
    return {"id": node_id, "parent": parent, "children": list(children), "message": message}


def chat():
    mapping = {
        "root": node("root", None, children=["sys"]),
        "sys": node("sys", "root", "system", {"content_type": "text", "parts": [""]}, children=["u1"]),
        "u1": node("u1", "sys", "user", {"content_type": "text", "parts": ["I decided to launch the beta in March."]},
                   children=["t1", "old"]),
        "old": node("old", "u1", "assistant", {"content_type": "text", "parts": ["ABANDONED BRANCH"]}),
        "t1": node("t1", "u1", "assistant", {"content_type": "thoughts", "thoughts": [{"summary": "s", "content": "HIDDEN THOUGHTS"}]},
                   children=["a1"]),
        "a1": node("a1", "t1", "assistant", {"content_type": "text", "parts": ["Great, March works."]}, children=["u2"]),
        "u2": node("u2", "a1", "user", {"content_type": "multimodal_text", "parts": [
            {"content_type": "audio_transcription", "text": "Spoken words from the user."},
            {"content_type": "image_asset_pointer", "asset_pointer": "file-service://file-abc"}]}),
    }
    return {"conversation_id": "gpt-1", "id": "gpt-1", "title": "Beta plan", "create_time": 1767225600.0,
            "update_time": 1767229200.5, "current_node": "u2", "mapping": mapping}


class ChatGPTTest(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.vault = self.root / "vault"
        (self.vault / "02-Conversations").mkdir(parents=True)
        self.export = self.root / "chatgpt_chats"
        self.export.mkdir()
        (self.export / "conversations-000.json").write_text(json.dumps([chat()]), encoding="utf-8")
        (self.export / "user.json").write_text('{"email": "PRIVATE"}', encoding="utf-8")
        self.state = self.root / "state"
        ingest.run([("chatgpt", self.export)], self.vault, state_dir=self.state)

    def note(self):
        return next((self.vault / "02-Conversations").glob("chatgpt-*.md")).read_text(encoding="utf-8")

    def test_active_branch_only_in_order(self):
        text = self.note()
        self.assertNotIn("ABANDONED BRANCH", text)
        self.assertLess(text.index("launch the beta"), text.index("March works"))
        self.assertLess(text.index("March works"), text.index("Spoken words"))
        self.assertIn("message_count: 4", text)

    def test_properties_and_extras(self):
        text = self.note()
        for expected in ('source_id: "gpt-1"', "source: chatgpt", "access: private",
                         'created_at_utc: "2026-01-01T00:00:00Z"', "### AI thinking", "HIDDEN THOUGHTS",
                         "### Media: image_asset_pointer", "## User", "## Assistant"):
            self.assertIn(expected, text)

    def test_archive_keeps_only_conversation_shards(self):
        archived = list((self.vault / "01-Raw-Exports" / "chatgpt").iterdir())
        self.assertEqual(len(archived), 1)
        self.assertEqual([p.name for p in archived[0].iterdir()], ["conversations-000.json"])
        self.assertEqual(ingest.run([("chatgpt", self.export)], self.vault, state_dir=self.state)
                         ["chatgpt", "unchanged"], 1)

    def test_extraction_sees_visible_text_only(self):
        calls = []

        def model(system, user):
            calls.append(user)
            return {"memories": [{"title": "Beta in March", "statement": "The user decided to launch the beta in March.",
                                  "category": "project_context", "origin": "user_stated", "message": 0,
                                  "quote": "decided to launch the beta in March", "project": None,
                                  "subject": "beta-launch", "flags": [], "revises": None}]}
        stats = extract.run(self.vault, model, "fake", state_dir=self.state)
        sent = "\n".join(calls)
        self.assertIn("Spoken words from the user.", sent)
        self.assertNotIn("HIDDEN THOUGHTS", sent)
        self.assertEqual(stats["created"], 1)
        memory = next((self.vault / "03-Memories").glob("*.md")).read_text(encoding="utf-8")
        self.assertIn('source_message_ids: ["u1"]', memory)
        self.assertIn("created: 2026-01-01", memory)


if __name__ == "__main__":
    unittest.main()
