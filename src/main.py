import argparse
import asyncio
import json
import os
from dotenv import load_dotenv

load_dotenv()

from src.ingestion import extract_questions_from_pdf
from src.agent import solve_all_questions


async def main():
    parser = argparse.ArgumentParser(description="Medical Coding Agent CLI")
    parser.add_argument("--pdf", type=str, default="data/practice_test_no_answers.pdf", help="Path to the exam PDF")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of questions to process")
    parser.add_argument("--output", type=str, default="exam_results.json", help="Output JSON file")
    parser.add_argument("--fresh", action="store_true", help="Start fresh, ignore existing results")
    parser.add_argument("--answers-csv", type=str, default="exam_answers.csv", help="Output CSV file for answers")

    args = parser.parse_args()

    print(f"Loading questions from {args.pdf}...")
    questions = extract_questions_from_pdf(args.pdf)
    print(f"Found {len(questions)} questions.")

    if args.limit:
        questions = questions[:args.limit]
        print(f"Limited to {len(questions)} questions.")

    # Get ordered question IDs for CSV output
    ordered_question_ids = list(range(1, len(questions) + 1))

    def write_answers_csv(results, path):
        if not path:
            return
        answer_map = {r["question_id"]: r.get("selected_option", "") for r in results}
        ordered_answers = [answer_map.get(qid, "") for qid in ordered_question_ids]
        with open(path, "w") as f:
            f.write(",".join(ordered_answers))

    def on_progress(msg):
        print(f"\r{msg}".ljust(80), end="", flush=True)

    print("\n--- Running Agent with Tool Use ---")
    print("Each question: Agent → lookup_codes tool → submit_answer tool\n")

    results = await solve_all_questions(questions, on_progress=on_progress)
    print("\n")

    # Save results
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {args.output}")

    write_answers_csv(results, args.answers_csv)
    print(f"Answers saved to {args.answers_csv}")

    # Summary
    success_count = sum(1 for r in results if r.get("confidence", 0) > 0)
    error_count = len(results) - success_count

    print(f"\n=== Summary ===")
    print(f"Answered: {success_count}/{len(results)}")
    if error_count:
        print(f"Errors:   {error_count}")


if __name__ == "__main__":
    asyncio.run(main())
