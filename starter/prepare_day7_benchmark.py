"""Create a Day 7 benchmark and agent cases from the Day 6 question set."""
import argparse
import json
from pathlib import Path
from typing import Any

from convert_day6 import convert_question


UNANSWERABLE = [
    {
        "id": "q21",
        "split": "dev",
        "question": (
            "What was the exact retail price in Vietnamese dong for the Biology 2e "
            "textbook on November 16, 2026?"
        ),
        "absence_reason": (
            "The indexed corpus contains Biology 2e sections 1.1–8.3, not current "
            "retail pricing. Confirm absence across the complete index before scoring."
        ),
    },
    {
        "id": "q22",
        "split": "dev",
        "question": "What room number hosted the BIO 101 lecture associated with this textbook?",
        "absence_reason": (
            "The corpus is textbook section content and does not specify course "
            "timetables or classroom assignments. Confirm absence across the complete index."
        ),
    },
    {
        "id": "q23",
        "split": "test",
        "question": (
            "What was the measured pH of the authors' private laboratory water sample "
            "on the day this section was written?"
        ),
        "absence_reason": (
            "The textbook sections do not document private author laboratory records. "
            "Confirm absence across the complete index."
        ),
    },
    {
        "id": "q24",
        "split": "test",
        "question": (
            "Which office and phone number should a student contact to change their "
            "biology lecture enrollment?"
        ),
        "absence_reason": (
            "The indexed textbook sections do not contain course enrollment contacts. "
            "Confirm absence across the complete index."
        ),
    },
]

FRESH_TEST_QUESTIONS = [
    {
        "id": "q16",
        "split": "test",
        "question": "Which groups of organisms are capable of photosynthesis?",
        "expected_answer": "Plants, algae, and cyanobacteria.",
        "gold_source_id": "biology-2e-section-8-1-overview-of-photosynthesis",
        "gold_page": 1,
        "evidence_quote": (
            "Plants, algae, and a group of bacteria called cyanobacteria are the "
            "only organisms capable of performing photosynthesis"
        ),
    },
    {
        "id": "q17",
        "split": "test",
        "question": "Where does glycolysis take place in both prokaryotic and eukaryotic cells?",
        "expected_answer": "In the cytoplasm.",
        "gold_source_id": "biology-2e-section-7-2-glycolysis",
        "gold_page": 1,
        "evidence_quote": (
            "Glycolysis takes place in the cytoplasm of both prokaryotic and "
            "eukaryotic cells."
        ),
    },
    {
        "id": "q18",
        "split": "test",
        "question": "Where do light-independent reactions of photosynthesis take place?",
        "expected_answer": "In the stroma of the chloroplast.",
        "gold_source_id": (
            "biology-2e-section-8-3-using-light-energy-to-make-organic-molecules"
        ),
        "gold_page": 1,
        "evidence_quote": "CO2 diffuses into the stroma of the chloroplast",
    },
    {
        "id": "q19",
        "split": "test",
        "question": "What is another name for the Calvin cycle that includes a second scientist?",
        "expected_answer": "The Calvin-Benson cycle.",
        "gold_source_id": (
            "biology-2e-section-8-3-using-light-energy-to-make-organic-molecules"
        ),
        "gold_page": 1,
        "evidence_quote": (
            "Others call it the Calvin-Benson cycle to include the name of another "
            "scientist involved in its discovery."
        ),
    },
    {
        "id": "q20",
        "split": "test",
        "question": "What gas supplies the carbon used to build carbohydrate molecules?",
        "expected_answer": "Carbon dioxide.",
        "gold_source_id": (
            "biology-2e-section-8-3-using-light-energy-to-make-organic-molecules"
        ),
        "gold_page": 1,
        "evidence_quote": "It comes from carbon dioxide",
    },
]


def prepare(raw_questions: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    answerable_dev = [
        question for question in raw_questions if question.get("split") == "dev"
    ]
    questions = [
        convert_question(question) for question in answerable_dev
    ] + [convert_question(question) for question in FRESH_TEST_QUESTIONS]
    ids = {question["id"] for question in questions}
    if len(ids) != len(questions):
        raise ValueError("Day 6 question IDs must be unique.")
    for item in UNANSWERABLE:
        if item["id"] in ids:
            raise ValueError(f"Duplicate benchmark question id: {item['id']}")
    if len(answerable_dev) < 15:
        raise ValueError(
            "Day 7 requires at least 15 answerable dev questions from Day 6."
        )
    answerable_cases = [
        {
            "id": question["id"],
            "split": question["split"],
            "answerable": True,
            "question": question["question"],
            **({"filter": question["filter"]} if "filter" in question else {}),
        }
        for question in questions
    ]
    unanswerable_cases = [
        {**item, "answerable": False} for item in UNANSWERABLE
    ]
    return questions, answerable_cases + unanswerable_cases


def validate_evidence(questions: list[dict[str, Any]], pages_path: Path) -> None:
    pages = {}
    with pages_path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue
            page = json.loads(line)
            key = (page["document_id"], int(page.get("page", 1)))
            if key in pages:
                raise ValueError(f"Duplicate Day 6 page at line {line_number}: {key}.")
            pages[key] = page["text"]

    for question in questions:
        for evidence in question["gold"]:
            key = (evidence["source_id"], int(evidence["page"]))
            text = pages.get(key)
            quote = evidence["evidence"]
            if text is None or quote not in text:
                raise ValueError(
                    f"{question['id']}: evidence quote not found in Day 6 page {key}."
                )
            start = 0
            while start < len(text):
                end = min(start + 500, len(text))
                if quote in text[start:end].strip():
                    break
                if end == len(text):
                    raise ValueError(
                        f"{question['id']}: evidence quote is split across Day 6 chunks."
                    )
                start = end - 50


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build Day 7 questions and agent cases from Day 6 questions."
    )
    parser.add_argument("--questions", required=True, type=Path)
    parser.add_argument("--pages", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    raw_questions = json.loads(args.questions.read_text(encoding="utf-8"))
    if not isinstance(raw_questions, list):
        raise ValueError("Day 6 questions must be a JSON array.")
    questions, cases = prepare(raw_questions)
    validate_evidence(questions, args.pages)
    args.output.mkdir(parents=True, exist_ok=True)
    for name, value in (("questions.json", questions), ("cases.json", cases)):
        (args.output / name).write_text(
            json.dumps(value, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    print(
        f"Created {len(questions)} answerable retrieval questions and "
        f"{len(cases) - len(questions)} unanswerable agent cases."
    )
    print("Human-check the evidence and verify each absence against the full index.")


if __name__ == "__main__":
    main()
