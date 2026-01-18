"""
Evaluation Script

Compares agent results against the answer key.
"""

import argparse
import json


def load_answers_csv(path: str) -> dict[int, str]:
    """Load answers from a comma-separated CSV (one row)."""
    with open(path) as f:
        content = f.read().strip().rstrip(".")
    if not content:
        return {}
    answers = [a.strip().upper() for a in content.split(",")]
    return {i + 1: ans for i, ans in enumerate(answers) if ans}


def load_results_json(path: str) -> dict[int, str]:
    """Load agent results from JSON file."""
    with open(path) as f:
        data = json.load(f)
    return {
        item["question_id"]: item.get("selected_option", "").upper()
        for item in data
        if item.get("question_id") and item.get("selected_option")
    }


def evaluate(answer_key: dict[int, str], results: dict[int, str]) -> None:
    """Compare results against answer key and print summary."""
    correct = incorrect = missing = 0

    print("\n=== Evaluation Results ===\n")

    for q_num in sorted(answer_key.keys()):
        expected = answer_key[q_num]
        actual = results.get(q_num)

        if actual is None:
            missing += 1
            print(f"Q{q_num}: MISSING (expected {expected})")
        elif actual == expected:
            correct += 1
        else:
            incorrect += 1
            print(f"Q{q_num}: X {actual} (expected {expected})")

    total = len(answer_key)
    answered = correct + incorrect

    print(f"\n=== Summary ===")
    print(f"Correct:   {correct}/{total} ({100*correct/total:.1f}%)")
    print(f"Incorrect: {incorrect}/{total} ({100*incorrect/total:.1f}%)")
    print(f"Missing:   {missing}/{total}")
    if answered > 0:
        print(f"Accuracy:  {100*correct/answered:.1f}% (of answered)")


def main():
    parser = argparse.ArgumentParser(description="Evaluate exam results")
    parser.add_argument("--answers", default="data/answers.csv", help="Answer key CSV")
    parser.add_argument("--results", default="exam_results.json", help="Results JSON")
    parser.add_argument("--results-csv", help="Results CSV (alternative to JSON)")
    args = parser.parse_args()

    print(f"Loading answer key from {args.answers}...")
    answer_key = load_answers_csv(args.answers)
    print(f"Found {len(answer_key)} answers.")

    if args.results_csv:
        print(f"Loading results from {args.results_csv}...")
        results = load_answers_csv(args.results_csv)
    else:
        print(f"Loading results from {args.results}...")
        results = load_results_json(args.results)
    print(f"Found {len(results)} results.")

    evaluate(answer_key, results)


if __name__ == "__main__":
    main()
