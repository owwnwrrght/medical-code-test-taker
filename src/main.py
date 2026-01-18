"""
Medical Coding Agent CLI

Run CPC exam questions through an AI agent with tool use.
"""

import argparse
import asyncio
import json

from dotenv import load_dotenv

load_dotenv()

from src.ingestion import extract_questions_from_pdf
from src.agent import solve_all_questions


async def main():
    parser = argparse.ArgumentParser(description="Medical Coding Agent")
    parser.add_argument("--pdf", default="data/practice_test_no_answers.pdf", help="Exam PDF path")
    parser.add_argument("--limit", type=int, help="Limit number of questions")
    parser.add_argument("--output", default="exam_results.json", help="Output JSON file")
    parser.add_argument("--answers-csv", default="exam_answers.csv", help="Output CSV file")
    args = parser.parse_args()

    # Load questions
    print(f"Loading questions from {args.pdf}...")
    questions = extract_questions_from_pdf(args.pdf)
    print(f"Found {len(questions)} questions.")

    if args.limit:
        questions = questions[:args.limit]
        print(f"Limited to {len(questions)} questions.")

    # Run agent
    print("\n--- Running Agent with Tool Use ---")
    print("Each question: Agent -> lookup_codes -> submit_answer\n")

    def on_progress(msg):
        print(f"\r{msg}".ljust(80), end="", flush=True)

    results = await solve_all_questions(questions, on_progress=on_progress)
    print("\n")

    # Save results
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {args.output}")

    # Save CSV
    answer_map = {r["question_id"]: r.get("selected_option", "") for r in results}
    ordered = [answer_map.get(i, "") for i in range(1, len(questions) + 1)]
    with open(args.answers_csv, "w") as f:
        f.write(",".join(ordered))
    print(f"Answers saved to {args.answers_csv}")

    # Summary
    answered = sum(1 for r in results if r.get("confidence", 0) > 0)
    print(f"\n=== Summary ===")
    print(f"Answered: {answered}/{len(results)}")


if __name__ == "__main__":
    asyncio.run(main())
