import unittest

import classify_projects as C


def row(title, text="details"):
    return (None, None, {"title": title, "source": "chatgpt", "id": title,
                         "messages": [{"sender": "user", "message": text, "_id": "m"}]})


ROWS = [row("Agent eval harness"), row("Eval lab metrics"), row("Sleep app icon"),
        row("Sleep app pricing"), row("Random trivia")]
LABELS = ["northwind", "northwind", "sleepcoach", "sleepcoach", None]
DISPLAY = {"northwind": "Northwind", "sleepcoach": "Sleep Coach"}


class ClassifyTest(unittest.TestCase):
    def test_prompt_excludes_tested_conversation(self):
        prompt = C.build_prompt(ROWS[0][2], ROWS, LABELS, DISPLAY, exclude=0)
        projects_part = prompt.split("New conversation title:")[0]
        self.assertNotIn("Agent eval harness", projects_part)
        self.assertIn("Eval lab metrics", projects_part)
        self.assertIn("Project: Sleep Coach", projects_part)
        self.assertNotIn("Random trivia", projects_part)

    def test_interpret(self):
        self.assertEqual(C.interpret({"project": "sleep coach", "confidence": "High"}, DISPLAY),
                         ("sleepcoach", "high"))
        self.assertEqual(C.interpret({"project": "None", "confidence": "low"}, DISPLAY), ("none", "low"))
        self.assertEqual(C.interpret({"project": "Made Up", "confidence": "high"}, DISPLAY)[0], None)
        self.assertEqual(C.interpret("text", DISPLAY), (None, ""))

    def test_sample_skips_singletons_and_unlabeled(self):
        labels = ["a"] * 30 + ["b"] * 10 + ["c"] + [None] * 5
        chosen = C.sample_indices(labels, size=8)
        self.assertTrue(all(labels[i] in ("a", "b") for i in chosen))
        self.assertIn("b", {labels[i] for i in chosen})
        self.assertEqual(chosen, C.sample_indices(labels, size=8))


class SuggestTest(unittest.TestCase):
    def test_writes_only_where_llm_and_embeddings_agree(self):
        import json, tempfile
        from pathlib import Path
        from unittest import mock
        import ingest
        with tempfile.TemporaryDirectory() as temp:
            vault = Path(temp)
            folder = vault / "02-Conversations"
            folder.mkdir()
            notes = {}
            for name, project in (("a", "Northwind"), ("b", "Northwind"), ("c", "Sleep Coach"),
                                  ("d", "Sleep Coach"), ("x", ""), ("y", ""), ("z", "")):
                notes[name] = folder / f"{name}.md"
                notes[name].write_text(f"---\nsource: chatgpt\nsource_id: {name}\nproject: {project}\n---\nbody\n",
                                       encoding="utf-8")
            rows = [(notes[n], ingest.split_note(notes[n].read_text())[0],
                     {"title": n, "source": "chatgpt", "id": n, "messages": []}) for n in "abcdxyz"]
            vectors = {"a": [1, 0], "b": [1, 0], "c": [0, 1], "d": [0, 1], "x": [1, 0], "y": [0, 1], "z": [1, 0]}
            answers = {"x": "Northwind", "y": "Northwind", "z": "none"}
            client = lambda system, prompt: {"project": answers[prompt.split("New conversation title: ")[1][0]],
                                             "confidence": "high"}
            with mock.patch.object(C.suggest_projects, "load_conversations", return_value=rows), \
                 mock.patch.object(C.suggest_projects, "vectors_for", return_value=[vectors[n] for n in "abcdxyz"]), \
                 mock.patch.object(C, "STATE", vault / "state"):
                counts = C.suggest(vault, "fake", apply=True, client=client)
            self.assertIn('project_suggested: "Northwind"', notes["x"].read_text())
            self.assertNotIn("project_suggested", notes["y"].read_text())
            self.assertNotIn("project_suggested", notes["z"].read_text())
            self.assertEqual(counts["agreed: suggested"], 1)
            self.assertEqual(counts["disagreed: left for you"], 2)
            self.assertIn("project: Northwind", notes["a"].read_text())


class PromoteTest(unittest.TestCase):
    def test_promotes_matching_known_rows_only(self):
        import tempfile
        from pathlib import Path
        from unittest import mock
        with tempfile.TemporaryDirectory() as temp:
            vault = Path(temp)
            (vault / "02-Conversations").mkdir()
            (vault / "05-Reports").mkdir()
            for stem, project in (("n1", ""), ("n2", ""), ("n3", ""), ("n4", "Already"), ("n5", "")):
                (vault / "02-Conversations" / f"{stem}.md").write_text(
                    f"---\nsource: chatgpt\nproject: {project}\n---\nbody\n", encoding="utf-8")
            (vault / "05-Reports" / "Project Suggestions.md").write_text("""# Project Suggestions

## Suggested (1)

| Suggested project | Similarity | Conversation |
|---|---:|---|
| Northwind | 0.8 | [[n5\\|Should be ignored]] |

## Disagreed (4): candidates to label by hand

| LLM says | Embeddings say | Similarity | Conversation |
|---|---|---:|---|
| sleep coach | Sleep Coach | 0.5 | [[n1\\|Pricing \\| plans]] |
| Northwind | Sleep Coach | 0.4 | [[n2\\|Still different]] |
| Helthcanvas | Helthcanvas | 0.4 | [[n3\\|Typo]] |
| Northwind | Northwind | 0.4 | [[n4\\|Has project]] |
""", encoding="utf-8")
            display = {"northwind": "Northwind", "sleepcoach": "Sleep Coach"}
            with mock.patch.object(C, "labeled_rows", return_value=([], [], display)):
                counts = C.promote(vault, apply=True)
            read = lambda stem: (vault / "02-Conversations" / f"{stem}.md").read_text()
            self.assertIn('project_suggested: "Sleep Coach"', read("n1"))
            for stem in ("n2", "n3", "n4", "n5"):
                self.assertNotIn("project_suggested", read(stem))
            self.assertEqual(counts["promoted to suggested"], 1)
            self.assertEqual(counts["unknown project name (skipped)"], 1)
            self.assertEqual(counts["project already set (skipped)"], 1)
            self.assertEqual(counts["still different (skipped)"], 1)


if __name__ == "__main__":
    unittest.main()
