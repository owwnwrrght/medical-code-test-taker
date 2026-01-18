#!/usr/bin/env python3
"""
Evaluation script to compare agent results against the answer key.
Usage: python3 src/evaluate.py
"""
import json
import argparse


def _load_answers_list(csv_path: str) -> list[str]:
    """Load answers from a one-row comma-separated CSV."""
    with open(csv_path, "r") as f:
        content = f.read().strip()

    content = content.rstrip(".")
    if not content:
        return []

    return [a.strip().upper() for a in content.split(",")]


def load_expected_from_csv(csv_path: str) -> dict[int, str]:
    """Load expected answers from CSV file (comma-separated, one row)."""
    answers_list = _load_answers_list(csv_path)
    return {i + 1: ans for i, ans in enumerate(answers_list) if ans}


def load_results(results_path: str) -> dict[int, str]:
    """Load agent results from JSON."""
    with open(results_path, "r") as f:
        data = json.load(f)
    
    results = {}
    for item in data:
        q_id = item.get("question_id")
        answer = item.get("selected_option", "").upper()
        if q_id and answer:
            results[q_id] = answer
    
    return results


def load_results_from_csv(csv_path: str) -> dict[int, str]:
    """Load agent results from a one-row comma-separated CSV."""
    answers_list = _load_answers_list(csv_path)
    return {i + 1: ans for i, ans in enumerate(answers_list) if ans}


def evaluate(answer_key: dict[int, str], results: dict[int, str]) -> None:
    """Compare results against answer key and print summary."""
    correct = 0
    incorrect = 0
    missing = 0
    
    print("\n=== Evaluation Results ===\n")
    
    for q_num in sorted(answer_key.keys()):
        expected = answer_key[q_num]
        actual = results.get(q_num)
        
        if actual is None:
            missing += 1
            print(f"Q{q_num}: MISSING (expected {expected})")
        elif actual == expected:
            correct += 1
            # print(f"Q{q_num}: ✓ {actual}")  # Uncomment for verbose output
        else:
            incorrect += 1
            print(f"Q{q_num}: ✗ {actual} (expected {expected})")
    
    total = len(answer_key)
    answered = correct + incorrect
    
    print(f"\n=== Summary ===")
    print(f"Correct:   {correct}/{total} ({100*correct/total:.1f}%)")
    print(f"Incorrect: {incorrect}/{total} ({100*incorrect/total:.1f}%)")
    print(f"Missing:   {missing}/{total}")
    print(f"Accuracy:  {100*correct/answered:.1f}% (of answered)")


def main():
    parser = argparse.ArgumentParser(description="Evaluate exam results against answer key")
    parser.add_argument("--answers", type=str, default="data/answers.csv", help="Path to answer key CSV")
    parser.add_argument("--results", type=str, default="exam_results.json", help="Path to results JSON")
    parser.add_argument("--results-csv", type=str, default=None, help="Path to results CSV")
    args = parser.parse_args()
    
    print(f"Loading answer key from {args.answers}...")
    answer_key = load_expected_from_csv(args.answers)
    print(f"Found {len(answer_key)} answers in key.")
    
    if args.results_csv:
        print(f"Loading results from {args.results_csv}...")
        results = load_results_from_csv(args.results_csv)
    else:
        print(f"Loading results from {args.results}...")
        results = load_results(args.results)
    print(f"Found {len(results)} answers in results.")
    
    evaluate(answer_key, results)



if __name__ == "__main__":
    main()
