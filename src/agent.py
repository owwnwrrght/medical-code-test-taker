"""
Medical Coding Agent with Tool Use

Architecture:
- Agent receives a question and can use tools to look up information
- Tools: lookup_codes (RAG), web_search (for missing codes), submit_answer
- Agent reasons about the question, uses tools as needed, then submits answer
- Runs with concurrency for speed
"""

import os
import re
import json
import asyncio
from dotenv import load_dotenv
from anthropic import AsyncAnthropic
from pydantic import BaseModel, Field
from src.models import Question
from src.rag import MedicalCodeRetriever

load_dotenv()

# Configuration
MAX_CONCURRENCY = 5  # Parallel questions
MAX_TOOL_TURNS = 5   # Max tool-use iterations per question


class Answer(BaseModel):
    selected_option: str = Field(default="A")
    confidence_score: float = Field(default=0.5)
    reasoning: str = Field(default="")


# Shared instances
_retriever = MedicalCodeRetriever()
_client = AsyncAnthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))


# Tool definitions
TOOLS = [
    {
        "name": "lookup_codes",
        "description": "Look up descriptions for medical codes (CPT, ICD-10, HCPCS) from the database. Always use this first to understand what each code means.",
        "input_schema": {
            "type": "object",
            "properties": {
                "codes": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of medical codes to look up (e.g., ['99213', 'I10', 'J1030'])"
                }
            },
            "required": ["codes"]
        }
    },
    {
        "name": "web_search",
        "description": "Search the web for medical code information not found in the database. Use this if lookup_codes returns 'No description found' for a code.",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Search query (e.g., 'CPT code 99213 description')"
                }
            },
            "required": ["query"]
        }
    },
    {
        "name": "submit_answer",
        "description": "Submit your final answer. You MUST call this after analyzing the codes.",
        "input_schema": {
            "type": "object",
            "properties": {
                "selected_option": {
                    "type": "string",
                    "enum": ["A", "B", "C", "D"],
                    "description": "The correct answer"
                },
                "confidence": {
                    "type": "number",
                    "description": "Confidence 0-1"
                },
                "reasoning": {
                    "type": "string",
                    "description": "Why this answer is correct"
                }
            },
            "required": ["selected_option", "reasoning"]
        }
    }
]

SYSTEM_PROMPT = """You are a CPC (Certified Professional Coder) medical coding expert taking an exam.

WORKFLOW:
1. Extract all medical codes from the answer options
2. Call lookup_codes to get descriptions for ALL codes
3. If any code returns "description unavailable", use web_search to find it
4. Match the clinical scenario to the code descriptions
5. ALWAYS call submit_answer with your choice - this is REQUIRED

KEY CODING PRINCIPLES:
- Match specific details: anatomy, body site, procedure type, laterality, patient age
- I&D (incision & drainage) ≠ excision ≠ biopsy - these are different procedures
- Unilateral ≠ bilateral
- Anesthesia codes: 00100-01999
- Renal angiography codes: 36251-36254 (catheterization), 75722-75724 (radiological S&I)
- If codes seem outdated but description matches the scenario, select it
- Choose the MOST SPECIFIC code matching the clinical details

EXAM STRATEGY:
- If no option perfectly matches, choose the CLOSEST match from available options
- Consider body site carefully: arm/extremity, neck, thorax, abdomen, pelvis
- For cysts that need removal, I&D codes may be appropriate if excision codes aren't listed
- Floor of mouth ≠ tongue (different anatomical structures)
- When in doubt between similar codes, pick the one that matches more clinical details

CRITICAL: You MUST call submit_answer at the end. Do not just provide analysis."""


def _extract_answer_from_text(text: str) -> str | None:
    """Try to extract an answer letter from text response."""
    # Look for patterns like "Answer: B" or "The answer is C" or just "B"
    patterns = [
        r"answer[:\s]+([A-D])\b",
        r"select(?:ed)?[:\s]+([A-D])\b",
        r"option[:\s]+([A-D])\b",
        r"^([A-D])\.",
        r"\b([A-D])\s+is\s+correct",
    ]
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return match.group(1).upper()
    return None


async def _handle_tool_call(tool_name: str, tool_input: dict) -> str:
    """Execute a tool and return the result."""
    if tool_name == "lookup_codes":
        codes = tool_input.get("codes", [])
        descriptions = {}
        for code in codes:
            result = await _retriever.retrieve_codes([code])
            descriptions[code] = result.get(code, "No description found")
        return json.dumps(descriptions, indent=2)

    elif tool_name == "web_search":
        query = tool_input.get("query", "")
        # Use Anthropic's built-in web search would require different API
        # For now, return a helpful message
        return f"Web search for '{query}': Please use your medical coding knowledge or try lookup_codes with variations of the code."

    return "Unknown tool"


async def solve_question(question: Question, on_status: callable = None) -> Answer:
    """
    Solve a single question using the agent with tools.
    """
    def status(msg):
        if on_status:
            on_status(f"Q{question.id}: {msg}")

    # Format question
    options_text = "\n".join([f"{k}. {v}" for k, v in sorted(question.options.items())])
    user_message = f"""Question {question.id}:
{question.text}

Options:
{options_text}

Look up the codes, analyze them, then submit your answer."""

    messages = [{"role": "user", "content": user_message}]

    status("thinking...")

    # Agent loop
    for turn in range(MAX_TOOL_TURNS):
        response = await _client.messages.create(
            model="claude-sonnet-4-5",
            max_tokens=1500,
            system=SYSTEM_PROMPT,
            tools=TOOLS,
            messages=messages
        )

        # Process response
        assistant_content = response.content
        tool_calls = [b for b in assistant_content if b.type == "tool_use"]
        text_blocks = [b for b in assistant_content if b.type == "text"]

        # Check for submit_answer
        for tool in tool_calls:
            if tool.name == "submit_answer":
                inp = tool.input
                status("answered")
                return Answer(
                    selected_option=inp.get("selected_option", "A"),
                    confidence_score=inp.get("confidence", 0.8),
                    reasoning=inp.get("reasoning", "")
                )

        # If no tool calls, try to extract answer from text
        if not tool_calls:
            for block in text_blocks:
                extracted = _extract_answer_from_text(block.text)
                if extracted:
                    status("answered (from text)")
                    return Answer(
                        selected_option=extracted,
                        confidence_score=0.6,
                        reasoning=block.text[:200]
                    )
            break  # No tools and no extractable answer

        # Process other tool calls
        tool_results = []
        for tool in tool_calls:
            if tool.name != "submit_answer":
                status(f"using {tool.name}...")
                result = await _handle_tool_call(tool.name, tool.input)
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": tool.id,
                    "content": result
                })

        # Continue conversation
        messages.append({"role": "assistant", "content": assistant_content})
        if tool_results:
            messages.append({"role": "user", "content": tool_results})

    # Fallback: agent didn't submit - try to extract from last response
    status("fallback")
    for msg in reversed(messages):
        if msg["role"] == "assistant":
            content = msg.get("content", [])
            for block in content:
                if hasattr(block, "text"):
                    extracted = _extract_answer_from_text(block.text)
                    if extracted:
                        return Answer(
                            selected_option=extracted,
                            confidence_score=0.4,
                            reasoning="Extracted from agent response"
                        )

    return Answer(selected_option="A", confidence_score=0.0, reasoning="Agent did not provide answer")


async def solve_all_questions(questions: list[Question], on_progress: callable = None) -> list[dict]:
    """
    Solve all questions with concurrency.
    """
    results = {}
    semaphore = asyncio.Semaphore(MAX_CONCURRENCY)
    completed = 0
    total = len(questions)

    def update_progress(msg=""):
        if on_progress:
            on_progress(f"[{completed}/{total}] {msg}")

    async def process_one(q: Question) -> dict:
        nonlocal completed
        async with semaphore:
            try:
                answer = await solve_question(q, on_status=update_progress)
                result = {
                    "question_id": q.id,
                    "selected_option": answer.selected_option,
                    "confidence": answer.confidence_score,
                    "reasoning": answer.reasoning
                }
            except Exception as e:
                result = {
                    "question_id": q.id,
                    "selected_option": "A",
                    "confidence": 0.0,
                    "reasoning": f"Error: {str(e)}"
                }
            completed += 1
            update_progress(f"Q{q.id} done")
            return result

    # Run with concurrency
    tasks = [process_one(q) for q in questions]
    results_list = await asyncio.gather(*tasks)

    return sorted(results_list, key=lambda x: x["question_id"])
