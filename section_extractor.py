"""
Section Extractor Module.
Uses LLM to extract structured sections from raw tender text.
Returns parsed JSON with 5 key sections.
"""

import json
from langchain_groq import ChatGroq
from prompts import SECTION_EXTRACTION_PROMPT, SUMMARY_PROMPT


def extract_sections(raw_text: str, llm: ChatGroq) -> dict:
    """
    Extract 5 structured sections from tender text using LLM.
    
    Returns dict with keys: scope, eligibility, emd, dates, documents
    """
    # Truncate if too long for context window
    text_for_extraction = raw_text[:12000] if len(raw_text) > 12000 else raw_text
    
    prompt = SECTION_EXTRACTION_PROMPT.format(text=text_for_extraction)
    
    try:
        response = llm.invoke(prompt)
        content = response.content.strip()
        
        # Clean up response — remove markdown code fences if present
        if content.startswith("```"):
            content = content.split("\n", 1)[1]  # Remove first line
            if content.endswith("```"):
                content = content[:-3]
            content = content.strip()
        
        sections = json.loads(content)
        
        # Validate expected keys exist
        expected_keys = ["scope", "eligibility", "emd", "dates", "documents"]
        for key in expected_keys:
            if key not in sections:
                sections[key] = "Not specified in the document."
        
        return sections
        
    except json.JSONDecodeError:
        # If LLM returns non-JSON, wrap the raw response
        return {
            "scope": content if content else "Could not extract scope.",
            "eligibility": "Could not extract — please check the Summary tab.",
            "emd": "Could not extract — please check the Summary tab.",
            "dates": "Could not extract — please check the Summary tab.",
            "documents": "Could not extract — please check the Summary tab."
        }
    except Exception as e:
        return {
            "scope": f"Error during extraction: {str(e)}",
            "eligibility": "Error",
            "emd": "Error",
            "dates": "Error",
            "documents": "Error"
        }


def generate_summary(raw_text: str, llm: ChatGroq) -> str:
    """
    Generate a plain-language summary of the tender document.
    Returns formatted markdown text.
    """
    text_for_summary = raw_text[:12000] if len(raw_text) > 12000 else raw_text
    prompt = SUMMARY_PROMPT.format(text=text_for_summary)
    
    try:
        response = llm.invoke(prompt)
        return response.content.strip()
    except Exception as e:
        return f"Error generating summary: {str(e)}"
