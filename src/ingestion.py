import fitz
import re
from typing import List
from src.models import Question

def extract_questions_from_pdf(pdf_path: str) -> List[Question]:
    """
    Extracts questions and options from the practice test PDF.
    """
    doc = fitz.open(pdf_path)
    full_text = ""
    
    # Skip intro pages (0 and 1) and maybe page 2 (timer)
    # Based on analysis, questions start on Page 4 (index 3)
    # But let's just process all pages and filter out non-questions later or rely on the regex
    
    for page in doc:
        text = page.get_text("text")
        full_text += text + "\n"
        
    # Remove headers/footers
    full_text = full_text.replace("Medical Coding Ace", "")
    full_text = full_text.replace("TIMER START", "")
    full_text = full_text.replace("4 HOURS", "")
    
    lines = full_text.split('\n')
    questions = []
    
    current_q_id = None
    current_q_text = []
    current_options = {}
    
    # Regex for Question start: "1. ", "100. "
    q_start_pattern = re.compile(r'^(\d+)\.\s+(.*)')
    # Regex for Option: "A. ", "B. "
    opt_pattern = re.compile(r'^([A-D])\.\s+(.*)')
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
            
        q_match = q_start_pattern.match(line)
        opt_match = opt_pattern.match(line)
        
        if q_match:
            # New question found. Save previous if exists.
            if current_q_id is not None:
                # Assuming we have valid options.
                questions.append(Question(
                    id=current_q_id,
                    text=" ".join(current_q_text).strip(),
                    options=current_options
                ))
            
            current_q_id = int(q_match.group(1))
            current_q_text = [q_match.group(2)]
            current_options = {}
            
        elif opt_match:
            # Found an option
            opt_label = opt_match.group(1)
            opt_text = opt_match.group(2)
            current_options[opt_label] = opt_text
            
        else:
            # Continuation of text
            # Depending on state, append to question or the last option
            if current_options:
                # Append to last option
                last_key = sorted(current_options.keys())[-1]
                current_options[last_key] += " " + line
            elif current_q_id is not None:
                # Append to question text
                current_q_text.append(line)
    
    # Add the last question
    if current_q_id is not None:
        questions.append(Question(
            id=current_q_id,
            text=" ".join(current_q_text).strip(),
            options=current_options
        ))
        
    return questions

if __name__ == "__main__":
    qs = extract_questions_from_pdf("practice_test_no_answers.pdf")
    print(f"Extracted {len(qs)} questions.")
    print(qs[0].to_string())
    print("-" * 20)
    print(qs[-1].to_string())
