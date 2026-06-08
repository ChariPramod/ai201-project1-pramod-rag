"""Evaluation harness — runs the 5 questions from planning.md and prints a
markdown table ready to paste into the README evaluation section.

Each row shows the question, the expected answer (verbatim from planning.md),
the top-k retrieved chunks (source + cosine distance), and the system's actual
answer. Accuracy is intentionally NOT judged here — that is the human
reviewer's call.
"""

from generator import generate_response
from retriever import retrieve

EVAL_SET = [
    {
        "question": "What do r/berkeley posters say about taking CS 61A with John DeNero — is he the recommended instructor for the course?",
        "expected": "(planning.md Q1 — fill from reddit_cs61a_denero.txt) The consensus the threads actually state, with the reasons given.",
    },
    {
        "question": "Why do students say CS 70 feels artificially hard — harder than its content warrants?",
        "expected": "(planning.md Q2 — fill from reddit_cs70_difficulty.txt) The specific causes posters cite (e.g. exam style vs. lecture, pacing, staffing) — answer must list the reasons present in the docs.",
    },
    {
        "question": "What do students say about the Stat 134 Spring 2025 midterm?",
        "expected": "(planning.md Q3 — fill from the Stat 134 thread) The specific complaint/observation in that thread.",
    },
    {
        "question": "According to r/berkeley, which courses get named the hardest at Berkeley?",
        "expected": "(planning.md Q4 — fill from reddit_hardest_class.txt) The courses the threads actually name — checkable as a list.",
    },
    {
        "question": "What is the average starting salary for CS 161 graduates?",
        "expected": "(planning.md Q5, out-of-scope) The system declines: \"I don't have enough information on that.\"",
    },
]


def _cell(s: str) -> str:
    """Make a string safe to drop into one cell of a GitHub-flavored markdown table."""
    return s.replace("|", "\\|").replace("\n", "<br>").strip()


def main() -> None:
    print("| # | Question | Expected (from planning.md) | Top retrieved (source — distance) | Actual answer |")
    print("|---|---|---|---|---|")
    for i, item in enumerate(EVAL_SET, 1):
        chunks = retrieve(item["question"])
        result = generate_response(item["question"])
        retrieved = "<br>".join(
            f"{c['source']} — {c['distance']:.4f}" for c in chunks
        ) or "(none)"
        print(
            f"| {i} "
            f"| {_cell(item['question'])} "
            f"| {_cell(item['expected'])} "
            f"| {retrieved} "
            f"| {_cell(result['answer'])} |"
        )


if __name__ == "__main__":
    main()
