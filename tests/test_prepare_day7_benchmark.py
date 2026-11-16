import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "starter"))
from prepare_day7_benchmark import prepare


def day6_question(index, split):
    return {
        "id": "q%02d" % index,
        "split": split,
        "question": "Question %d?" % index,
        "expected_answer": "Answer %d" % index,
        "gold_source_id": "doc",
        "gold_page": 1,
        "evidence_quote": "evidence %d" % index,
    }


class PrepareBenchmarkTests(unittest.TestCase):
    def test_adds_four_unanswerable_cases_without_leaking_gold_to_agent_cases(self):
        raw_questions = [
            day6_question(index, "dev" if index <= 15 else "test")
            for index in range(1, 21)
        ]

        questions, cases = prepare(raw_questions)

        self.assertEqual(len(questions), 20)
        self.assertEqual(len(cases), 24)
        self.assertEqual(
            sum(case["answerable"] and case["split"] == "dev" for case in cases),
            15,
        )
        self.assertEqual(
            sum(case["answerable"] and case["split"] == "test" for case in cases),
            5,
        )
        self.assertEqual(
            sum(not case["answerable"] and case["split"] == "dev" for case in cases),
            2,
        )
        self.assertEqual(
            sum(not case["answerable"] and case["split"] == "test" for case in cases),
            2,
        )
        self.assertNotIn("gold", cases[0])
        self.assertNotIn("expected_answer", cases[0])
        self.assertTrue(all(case["absence_reason"] for case in cases if not case["answerable"]))


if __name__ == "__main__":
    unittest.main()
