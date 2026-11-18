from pathlib import Path
import tempfile
import unittest

import flag

RULES = flag.load_rules()


def note(statement, flags='flags: []', extra="", title="A memory"):
    return (f"---\nid: mem-1\ntype: memory\ntitle: \"{title}\"\n{flags}\n{extra}access: private\n---\n\n"
            f"{statement}\n\n## Evidence\n\nUser said:\n\n> quote\n\n## Source\n\n[[c|Cholesterol and my salary chat]]\n")


class FlagTest(unittest.TestCase):
    def flags(self, text):
        lines, _ = flag.ingest.split_note(text)
        return flag.read_list(lines, "flags")[0], flag.read_list(lines, "keyword_flags")[0]

    def test_patterns(self):
        cases = {
            "The user stored an api key sk-abcdefghijklmnopqrstuvwx in the repo.": {"credentials"},
            "The user's base salary offer is $185k plus equity.": {"financial"},
            "The user was prescribed a statin after high LDL.": {"health"},
            "The user's manager at the company approved the reorg.": {"employer"},
            "The user's manager approved the plan.": {"employer"},
            "The user's wife prefers Austin.": {"family"},
            "The user is moving to Seattle next spring.": {"location"},
            "The user saw their doctor last week.": {"health"},
            "The user prefers Postgres for relational data.": set(),
            "The user discussed a reorg with my manager.": {"employer"},
            "My wife and I are moving to Austin.": {"family", "location"},
            "The user is waiting on an H-1B transfer.": {"legal"},
        }
        for statement, expected in cases.items():
            self.assertEqual(flag.match_flags(statement, RULES), expected, statement)

    def test_health_tech_is_not_personal_health(self):
        text = "The user is building a consumer health platform with generative UI for diabetes and asthma patients."
        self.assertEqual(flag.match_flags(text, RULES), set())

    def test_source_section_is_ignored(self):
        new, added = flag.flag_note(note("The user prefers Postgres."), RULES)
        self.assertEqual(added, set())

    def test_adds_flags_and_records_them(self):
        new, added = flag.flag_note(note("The user's salary is $150k.", flags='flags: ["health"]'), RULES)
        self.assertEqual(added, {"financial"})
        self.assertEqual(self.flags(new), (["health", "financial"], ["financial"]))
        self.assertIn("The user's salary is $150k.", new)

    def test_handles_obsidian_block_lists(self):
        text = note("The user's salary is $150k.", flags="flags:\n  - family")
        new, added = flag.flag_note(text, RULES)
        self.assertEqual(self.flags(new)[0], ["family", "financial"])
        self.assertNotIn("  - family", new)

    def test_idempotent(self):
        once, _ = flag.flag_note(note("The user's salary is $150k."), RULES)
        twice, added = flag.flag_note(once, RULES)
        self.assertEqual(once, twice)
        self.assertEqual(added, set())

    def test_removed_flag_is_not_re_added(self):
        once, _ = flag.flag_note(note("The user's salary is $150k."), RULES)
        removed = once.replace('\nflags: ["financial"]', "\nflags: []")
        again, added = flag.flag_note(removed, RULES)
        self.assertEqual(again, removed)
        self.assertEqual(added, set())

    def test_existing_model_flag_is_recorded_without_change(self):
        text = note("The user's salary is $150k.", flags='flags: ["financial"]')
        new, added = flag.flag_note(text, RULES)
        self.assertEqual(added, set())
        self.assertEqual(new, text)

    def test_run_only_touches_memories(self):
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp)
            (folder / "mem-1.md").write_text(note("The user's salary is $150k."), encoding="utf-8")
            (folder / "Memories.base").write_text("views: []\n", encoding="utf-8")
            (folder / "other.md").write_text("---\ntype: conversation\n---\nsalary $1\n", encoding="utf-8")
            stats = flag.run(folder, RULES)
            self.assertEqual(stats["added: financial"], 1)
            self.assertEqual((folder / "other.md").read_text(encoding="utf-8"), "---\ntype: conversation\n---\nsalary $1\n")


if __name__ == "__main__":
    unittest.main()
