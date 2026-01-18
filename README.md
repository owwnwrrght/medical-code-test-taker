# Medical Coding Agent

An AI agent that takes CPC (Certified Professional Coder) medical coding exams using **tool use** and **RAG** (Retrieval-Augmented Generation).

## Overview

This project demonstrates an agentic approach to medical coding exams. Rather than stuffing all medical code information into a prompt, the agent actively queries a database of medical codes, reasons about the clinical scenario, and submits structured answers.

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                           AGENT LOOP                                │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│   Question ──► Agent (Gemini Flash)                                 │
│                    │                                                │
│                    ├──► Tool: lookup_codes(["99213", "I10"])        │
│                    │    └──► RAG: Query medical code database       │
│                    │         └──► Returns: Code descriptions        │
│                    │                                                │
│                    ├──► Agent reasons about codes vs question       │
│                    │                                                │
│                    └──► Tool: submit_answer("C", reasoning)         │
│                         └──► Returns structured answer              │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

## Tools

| Tool | Description | Input | Output |
|------|-------------|-------|--------|
| `lookup_codes` | Query the medical code database (CPT, ICD-10, HCPCS) | List of codes | Code descriptions |
| `submit_answer` | Submit final answer with reasoning | Option + reasoning | Structured answer |

## Data Sources

- **SQLite Database**: Pre-built from UMLS MRCONSO.RRF containing CPT, ICD-10, and HCPCS codes
- **ICD-10-CM Codes**: 2026 official code list
- **HCPCS Codes**: 2026 January release
- **Supplementary Codes**: Hand-curated CPT descriptions for common exam codes

## Quick Start

### Prerequisites

- Python 3.10+
- Gemini API key

### Installation

```bash
# Clone the repository
git clone https://github.com/owwnwrrght/medical-code-test-taker.git
cd medical-code-test-taker

# Install dependencies
pip install -r requirements.txt

# Set up environment variables
echo "GEMINI_API_KEY=your-key-here" > .env
```

### Setting Up Medical Code Data

The `codes/` directory is not included in this repository due to copyright restrictions. You'll need to provide your own medical code data:

1. Create a `codes/` directory
2. Add the following files:
   - `codes.sqlite` - SQLite database with CPT/ICD-10/HCPCS codes
   - `icd10cm-codes-2026.txt` - ICD-10-CM codes (from CMS)
   - `HCPC2026_JAN_ANWEB_01122026.txt` - HCPCS codes (from CMS)
   - `cpt_supplement.json` - Optional supplementary CPT descriptions

The ICD-10 and HCPCS files are freely available from [CMS.gov](https://www.cms.gov/). CPT codes require a license from the AMA.

### Usage

```bash
# Run the full exam
PYTHONPATH=. python3 src/main.py

# Run with a limit (for testing)
PYTHONPATH=. python3 src/main.py --limit 10

# Evaluate results against answer key
PYTHONPATH=. python3 src/evaluate.py --results-csv exam_answers.csv
```

### Command Line Options

| Option | Description | Default |
|--------|-------------|---------|
| `--pdf` | Path to exam PDF | `data/practice_test_no_answers.pdf` |
| `--limit` | Number of questions to process | All |
| `--output` | Output JSON file | `exam_results.json` |
| `--answers-csv` | Output CSV file | `exam_answers.csv` |

## Project Structure

```
Medical-Exam-Pro/
├── src/
│   ├── __init__.py      # Package exports
│   ├── agent.py         # Agent with tool use (main logic)
│   ├── main.py          # CLI entry point
│   ├── rag.py           # Medical code retrieval (RAG)
│   ├── models.py        # Data models
│   ├── ingestion.py     # PDF parsing
│   └── evaluate.py      # Accuracy evaluation
├── codes/
│   ├── codes.sqlite     # Pre-built code database
│   ├── cpt_supplement.json  # Supplementary CPT codes
│   ├── icd10cm-codes-2026.txt
│   ├── HCPC2026_JAN_ANWEB_01122026.txt
│   └── 2026_DHS_Code_List_Addendum_12_01_2025.txt
├── data/
│   ├── practice_test_no_answers.pdf  # Sample exam
│   └── answers.csv      # Answer key
├── requirements.txt     # Python dependencies
├── .gitignore
└── README.md
```

## Design Decisions

### 1. Tool Use Over Direct Prompting

Instead of embedding all code descriptions in the prompt, the agent actively queries for relevant information. This:
- Reduces token usage significantly
- Allows the agent to look up additional codes as needed
- Demonstrates true agentic behavior

The agent also computes per-option match scores from retrieved descriptions and only invokes the LLM when needed.
It applies a rule filter (procedure/site matching) before the LLM, and can auto-answer when only one option remains.
If exactly one option has a non-zero description match, it is treated as dominant and can override LLM picks; a small set of keyword rules handles under-specified anatomy.

### 2. RAG for Medical Codes

Medical codes are stored in a SQLite database with multiple lookup strategies:
1. Exact code match
2. Code prefix matching (for ICD-10 hierarchies)
3. Supplementary JSON file for common exam codes

### 3. Concurrent Processing

Questions are processed in parallel (default: 5 concurrent) using asyncio, respecting API rate limits while maximizing throughput.

### 4. Structured Output

The agent uses a `submit_answer` tool to return structured responses with:
- Selected option (A/B/C/D)
- Confidence score (0-1)
- Reasoning explanation

## Performance

- **Accuracy**: ~72% on 100-question CPC practice exam
- **Speed**: ~3-4 minutes for 100 questions (with concurrency)
- **API Calls**: ~100-200 total (1-2 per question depending on tool iterations)

## Configuration

Environment variables (set in `.env`):

| Variable | Description | Default |
|----------|-------------|---------|
| `GEMINI_API_KEY` | Gemini API key | Required |
| `GOOGLE_API_KEY` | Gemini API key alias | Optional |
| `GEMINI_MODEL` | Gemini model name | `gemini-1.5-flash` |
| `GEMINI_BASE_URL` | Gemini OpenAI-compatible base URL | `https://generativelanguage.googleapis.com/v1beta/openai/` |
| `LLM_MAX_RETRIES` | Max retry attempts | `3` |
| `LLM_RETRY_BASE_SECONDS` | Base retry delay | `1` |
| `LLM_RETRY_MAX_SECONDS` | Max retry delay | `8` |
| `MEDICAL_CODES_DIR` | Custom codes directory | `./codes` |
| `HEURISTIC_ENABLED` | Enable heuristic auto-answering | `true` |
| `HEURISTIC_MIN_SCORE` | Minimum heuristic score to auto-answer | `0.35` |
| `HEURISTIC_MIN_GAP` | Minimum score gap vs runner-up | `0.15` |
| `HEURISTIC_MIN_OPTIONS` | Minimum options with known descriptions | `3` |
| `HEURISTIC_DOMINANT_MIN_SCORE` | Minimum score when the top option is the only non-zero match | `0.08` |
| `EVIDENCE_ENABLED` | Add per-option evidence pack to prompt | `true` |
| `EVIDENCE_MAX_DESC_LEN` | Truncate code descriptions in evidence pack | `200` |
| `SECOND_PASS_ENABLED` | Enable second-pass re-evaluation | `true` |
| `SECOND_PASS_MIN_LLM_CONF` | LLM confidence threshold to trigger second pass | `0.55` |
| `SECOND_PASS_MIN_HEURISTIC_SCORE` | Heuristic score needed for second pass | `0.5` |
| `SECOND_PASS_MIN_HEURISTIC_GAP` | Heuristic gap needed for second pass | `0.15` |
| `SECOND_PASS_MIN_OPTIONS` | Minimum options with known descriptions | `3` |
| `DISAGREE_SECOND_PASS` | Trigger second pass on rule/heuristic disagreement | `true` |
| `RULE_FILTER_ENABLED` | Enable rule-based filtering before LLM | `true` |
| `RULE_FALLBACK_ENABLED` | Enable fallback rule when no procedure matches | `true` |
| `RULE_FALLBACK_AUTO` | Auto-answer using fallback rule | `true` |
| `KEYWORD_RULES_ENABLED` | Enable keyword-based disambiguation rules | `true` |
| `TRACE_ENABLED` | Include tool trace in results JSON | `true` |
| `TRACE_SCORE_PRECISION` | Decimal precision for trace scores | `3` |

## License

MIT
