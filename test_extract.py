import json
from pathlib import Path
import tempfile
import unittest

import extract
import ingest


def msg(uuid, sender, text, extra=()):
    return {"uuid": uuid, "sender": sender, "text": text, "created_at": "2026-05-01T10:00:00Z",
            "parent_message_uuid": None, "attachments": [], "files": [],
            "content": [{"type": "text", "text": text}, *extra]}


USER_LINE = "I have decided to build the backend in Rust for this product."
AI_LINE = "You should definitely move the whole system to Kubernetes next quarter."


def chat():
    return {"uuid": "conv-1", "name": "Stack chat", "created_at": "2026-05-01T10:00:00Z",
            "updated_at": "2026-05-01T11:00:00Z",
            "chat_messages": [
                msg("m0", "human", USER_LINE,
                    [{"type": "injected_prompt_block", "prompt": "INJECTED SECRET"}]),
                msg("m1", "assistant", AI_LINE,
                    [{"type": "thinking", "thinking": "THINKING SECRET"}]),
                msg("m2", "human", "Yes, I agree, let's go with Kubernetes then."),
            ]}


def memory(**changes):
    item = {"title": "Rust backend", "statement": "The user chose Rust.",
            "category": "technical_decision", "origin": "user_stated", "message": 0,
            "quote": "decided to build the backend in Rust", "project": "Stack",
            "subject": "backend language", "flags": [], "revises": None}
    item.update(changes)
    return item


class FakeModel:
    def __init__(self, memories):
        self.memories, self.calls = memories, []

    def __call__(self, system, user):
        self.calls.append(user)
        return {"memories": self.memories}


class ExtractTest(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.vault = self.root / "vault"
        (self.vault / "02-Conversations").mkdir(parents=True)
        self.state = self.root / "state"
        self.export = self.root / "conversations.json"
        self.export.write_text(json.dumps([chat()]), encoding="utf-8")
        ingest.run([("claude", self.export)], self.vault, state_dir=self.state)

    def extract(self, model):
        return extract.run(self.vault, model, "fake", state_dir=self.state)

    def notes(self):
        return sorted((self.vault / "03-Memories").glob("*.md"))

    def only_note(self):
        notes = self.notes()
        self.assertEqual(len(notes), 1)
        return notes[0].read_text(encoding="utf-8")

    def test_note_has_provenance_and_is_private(self):
        self.extract(FakeModel([memory()]))
        text = self.only_note()
        for expected in ("type: memory", "access: private", "attribution: user_stated",
                         'source_conversation_id: "claude-conv-1"', 'source_message_ids: ["m0"]',
                         "extracted_at:", "source_conversation: \"[[claude-", "User said:"):
            self.assertIn(expected, text)

    def test_invented_quote_is_rejected(self):
        stats = self.extract(FakeModel([memory(quote="I will quit my job and move abroad")]))
        self.assertEqual(self.notes(), [])
        self.assertEqual(stats["rejected: quote not found"], 1)

    def test_quote_from_a_different_message_is_rejected(self):
        self.extract(FakeModel([memory(message=1)]))
        self.assertEqual(self.notes(), [])

    def test_ai_recommendation_cannot_become_user_statement(self):
        stats = self.extract(FakeModel([memory(
            message=1, quote="move the whole system to Kubernetes", origin="user_stated")]))
        self.assertIn("attribution: ai_suggested", self.only_note())
        self.assertIn("AI said:", self.only_note())
        self.assertEqual(stats["attribution corrected"], 1)

    def test_confirmation_must_cite_user_after_ai_message(self):
        self.extract(FakeModel([memory(
            message=2, quote="let's go with Kubernetes", origin="user_confirmed_ai_suggestion")]))
        self.assertIn("attribution: user_confirmed_ai_suggestion", self.only_note())

    def test_confirmation_without_prior_ai_message_is_downgraded(self):
        self.extract(FakeModel([memory(origin="user_confirmed_ai_suggestion")]))
        self.assertIn("attribution: user_stated", self.only_note())

    def test_hidden_content_is_not_sent_to_model(self):
        model = FakeModel([])
        self.extract(model)
        sent = "\n".join(model.calls)
        self.assertIn(USER_LINE, sent)
        self.assertNotIn("INJECTED SECRET", sent)
        self.assertNotIn("THINKING SECRET", sent)

    def test_rerun_skips_and_keeps_review_edits(self):
        self.extract(FakeModel([memory()]))
        path = self.notes()[0]
        path.write_text(path.read_text(encoding="utf-8").replace(
            "access: private", "access: approved"), encoding="utf-8")
        model = FakeModel([memory()])
        stats = self.extract(model)
        self.assertEqual(model.calls, [])
        self.assertEqual(stats["conversations skipped (already extracted)"], 1)
        self.assertIn("access: approved", path.read_text(encoding="utf-8"))

    def test_changed_conversation_keeps_existing_memory(self):
        self.extract(FakeModel([memory()]))
        path = self.notes()[0]
        path.write_text(path.read_text(encoding="utf-8") + "\nReviewed.\n", encoding="utf-8")
        changed = chat()
        changed["chat_messages"].append(msg("m3", "human", "One more thought about it all."))
        for archived in (self.vault / "01-Raw-Exports" / "claude").glob("*.json"):
            archived.chmod(0o600)
            archived.write_text(json.dumps([changed]), encoding="utf-8")
        stats = self.extract(FakeModel([memory()]))
        self.assertEqual(stats["already existed"], 1)
        self.assertIn("Reviewed.", path.read_text(encoding="utf-8"))

    def test_revision_links_within_response(self):
        self.extract(FakeModel([
            memory(), memory(message=2, quote="let's go with Kubernetes", title="K8s", revises=0)]))
        texts = [p.read_text(encoding="utf-8") for p in self.notes()]
        self.assertEqual(sum('revises: "[[mem-' in text for text in texts), 1)

    def test_failed_conversation_stays_pending(self):
        def broken(system, user):
            raise ValueError("bad json")
        stats = self.extract(broken)
        self.assertEqual(stats["conversations failed"], 1)
        self.assertEqual(self.extract(FakeModel([]))["conversations processed"], 1)

    def test_long_message_is_fully_covered(self):
        items = [{"number": 0, "who": "user", "text": "a" * 40000 + "TAIL"}]
        joined = "".join(p[2] for chunk in extract.build_chunks(items) for p in chunk)
        self.assertIn("TAIL", joined)
        self.assertTrue(all(sum(len(p[2]) for p in c) <= extract.CHUNK_CHARS
                            for c in extract.build_chunks(items)))

    def test_rejects_remote_endpoint(self):
        with self.assertRaises(SystemExit):
            extract.LocalModel(url="https://api.example.com/v1/chat/completions")

    def test_parse_response_strips_reasoning(self):
        self.assertEqual(extract.parse_response('<think>{"x":1}</think>\n{"memories": []}'),
                         {"memories": []})

    def test_local_model_retries_once_on_non_json(self):
        model = extract.LocalModel()
        replies = ["```text\nnot json\n```", '{"memories": []}']
        sent = []
        model.send = lambda payload: (sent.append(payload["messages"][1]["content"]), replies.pop(0))[1]
        self.assertEqual(model("system", "transcript"), {"memories": []})
        self.assertEqual(sent[0], "transcript")
        self.assertIn("not a JSON object", sent[1])

    def test_local_model_gives_up_after_second_failure(self):
        model = extract.LocalModel()
        model.send = lambda payload: "still not json"
        with self.assertRaises(ValueError):
            model("system", "transcript")


if __name__ == "__main__":
    unittest.main()
