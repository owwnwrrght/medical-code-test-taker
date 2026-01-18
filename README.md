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
│   Question ──► Agent (Claude Sonnet 4.5)                            │
│                    │                                                │
│                    ├──► Tool: lookup_codes(["99213", "I10"])        │
│                    │    └──► RAG: Query medical code database       │
│                    │         └──► Returns: Code descriptions        │
│                    │                                                │
│                    ├──► Tool: web_search("CPT 99213")               │
│                    │    └──► Fallback for codes not in database     │
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
| `web_search` | Search for codes not in database | Search query | Code information |
| `submit_answer` | Submit final answer with reasoning | Option + reasoning | Structured answer |

## Data Sources

- **SQLite Database**: Pre-built from UMLS MRCONSO.RRF containing CPT, ICD-10, and HCPCS codes
- **ICD-10-CM Codes**: 2026 official code list
- **HCPCS Codes**: 2026 January release
- **Supplementary Codes**: Hand-curated CPT descriptions for common exam codes

## Quick Start

### Prerequisites

- Python 3.10+
- Anthropic API key

### Installation

```bash
# Clone the repository
git clone https://github.com/owwnwrrght/medical-code-test-taker.git
cd medical-code-test-taker

# Install dependencies
pip install -r requirements.txt

# Set up environment variables
echo "ANTHROPIC_API_KEY=your-key-here" > .env
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

### 2. RAG for Medical Codes

Medical codes are stored in a SQLite database with multiple lookup strategies:
1. Exact code match
2. Code prefix matching (for ICD-10 hierarchies)
3. Supplementary JSON file for common exam codes
4. AAPC web fallback (when enabled)

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
| `ANTHROPIC_API_KEY` | Anthropic API key | Required |
| `AAPC_LOOKUP` | Enable AAPC web scraping | `false` |
| `MEDICAL_CODES_DIR` | Custom codes directory | `./codes` |

## License

MIT
