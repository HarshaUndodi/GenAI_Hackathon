"""
Hybrid Eligibility Engine.
Step 1: LLM extracts eligibility criteria as structured JSON.
Step 2: Python rule engine compares vendor profile against criteria.
Step 3: Generates per-criterion pass/fail report with reasoning.

This is the KEY differentiator — NOT letting the LLM decide pass/fail.
"""

import json
import re
from langchain_groq import ChatGroq
from prompts import CRITERIA_EXTRACTION_PROMPT


def extract_criteria(raw_text: str, llm: ChatGroq) -> list:
    """
    Step 1: Use LLM to extract eligibility criteria as structured JSON.
    Returns a list of criterion dicts.
    """
    text_for_extraction = raw_text[:12000] if len(raw_text) > 12000 else raw_text
    prompt = CRITERIA_EXTRACTION_PROMPT.format(text=text_for_extraction)
    
    try:
        # Force JSON mode for 100% reliable parsing
        llm_json = llm.bind(response_format={"type": "json_object"})
        response = llm_json.invoke(prompt)
        content = response.content.strip()
        
        parsed = json.loads(content, strict=False)
        criteria = parsed.get("criteria", [])
        
        if not isinstance(criteria, list):
            return []
            
        return criteria
        
    except Exception as e:
        print(f"Extraction error: {e}")
        return []


def check_eligibility(criteria: list, vendor_profile: dict) -> list:
    """
    Step 2: Pure Python comparison of vendor profile against extracted criteria.
    No LLM involvement — deterministic rule-based evaluation.
    
    vendor_profile expected format:
    {
        "company_name": "Tech Corp",
        "turnover": 500.0,          # in lakhs
        "experience_years": 5,
        "projects_completed": 10,
        "msme_status": True,
        "iso_certified": True,
        "certifications": ["ISO 9001", "ISO 14001"]
    }
    """
    results = []
    
    for criterion in criteria:
        criterion_type = criterion.get("type", "text")
        field = criterion.get("field", "")
        operator = criterion.get("operator", ">=")
        required_value = criterion.get("value")
        criterion_name = criterion.get("criterion", "Unknown criterion")
        source_text = criterion.get("source_text", "")
        
        vendor_value = vendor_profile.get(field)
        result = {
            "criterion": criterion_name,
            "required": str(required_value),
            "vendor_value": str(vendor_value),
            "source_text": source_text,
            "status": "FAIL",
            "reasoning": ""
        }
        
        try:
            if criterion_type == "numeric":
                result = _check_numeric(
                    result, vendor_value, required_value, operator, field
                )
            elif criterion_type == "boolean":
                result = _check_boolean(
                    result, vendor_value, required_value, field
                )
            elif criterion_type == "text":
                result = _check_text(
                    result, vendor_value, required_value, vendor_profile, field
                )
            else:
                result["status"] = "REVIEW"
                result["reasoning"] = f"Unknown criterion type '{criterion_type}' — manual review needed."
                
        except Exception as e:
            result["status"] = "REVIEW"
            result["reasoning"] = f"Could not evaluate: {str(e)}"
        
        results.append(result)
    
    return results


def _check_numeric(result: dict, vendor_value, required_value, operator: str, field: str) -> dict:
    """Evaluate numeric criteria (turnover, experience, projects)."""
    if vendor_value is None:
        result["status"] = "REVIEW"
        result["reasoning"] = f"Vendor did not provide '{field}' information."
        return result
    
    try:
        vendor_num = float(vendor_value)
        required_num = float(required_value)
    except (ValueError, TypeError):
        result["status"] = "REVIEW"
        result["reasoning"] = f"Could not parse values for comparison. Vendor: {vendor_value}, Required: {required_value}"
        return result
    
    result["vendor_value"] = str(vendor_num)
    
    comparisons = {
        ">=": vendor_num >= required_num,
        "<=": vendor_num <= required_num,
        ">": vendor_num > required_num,
        "<": vendor_num < required_num,
        "==": vendor_num == required_num,
    }
    
    passed = comparisons.get(operator, vendor_num >= required_num)
    
    if passed:
        result["status"] = "PASS"
        result["reasoning"] = f"Vendor value ({vendor_num}) meets requirement ({operator} {required_num})."
    else:
        result["status"] = "FAIL"
        result["reasoning"] = f"Vendor value ({vendor_num}) does NOT meet requirement ({operator} {required_num})."
    
    return result


def _check_boolean(result: dict, vendor_value, required_value, field: str) -> dict:
    """Evaluate boolean criteria (MSME, ISO, etc.)."""
    if vendor_value is None:
        result["status"] = "REVIEW"
        result["reasoning"] = f"Vendor did not provide '{field}' information."
        return result
    
    vendor_bool = bool(vendor_value)
    
    # Parse required value
    if isinstance(required_value, bool):
        required_bool = required_value
    elif isinstance(required_value, str):
        required_bool = required_value.lower() in ("true", "yes", "1", "required", "mandatory")
    else:
        required_bool = bool(required_value)
    
    result["vendor_value"] = "Yes" if vendor_bool else "No"
    result["required"] = "Required" if required_bool else "Not Required"
    
    if required_bool and vendor_bool:
        result["status"] = "PASS"
        result["reasoning"] = f"Vendor has the required '{field}'."
    elif required_bool and not vendor_bool:
        result["status"] = "FAIL"
        result["reasoning"] = f"Vendor does NOT have the required '{field}'."
    else:
        result["status"] = "PASS"
        result["reasoning"] = f"'{field}' is not mandatory — vendor status acceptable."
    
    return result


def _check_text(result: dict, vendor_value, required_value, vendor_profile: dict, field: str) -> dict:
    """Evaluate text-based criteria (specific certifications, etc.)."""
    if vendor_value is None:
        # Check if it might be in certifications list
        certs = vendor_profile.get("certifications", [])
        if isinstance(required_value, str) and any(
            required_value.lower() in c.lower() for c in certs
        ):
            result["status"] = "PASS"
            result["vendor_value"] = ", ".join(certs)
            result["reasoning"] = f"Required '{required_value}' found in vendor certifications."
            return result
        
        result["status"] = "REVIEW"
        result["reasoning"] = f"Vendor did not provide '{field}' information."
        return result
    
    if isinstance(vendor_value, list):
        if isinstance(required_value, str) and any(
            required_value.lower() in str(v).lower() for v in vendor_value
        ):
            result["status"] = "PASS"
            result["vendor_value"] = ", ".join(str(v) for v in vendor_value)
            result["reasoning"] = f"Required '{required_value}' found in vendor data."
        else:
            result["status"] = "FAIL"
            result["vendor_value"] = ", ".join(str(v) for v in vendor_value)
            result["reasoning"] = f"Required '{required_value}' NOT found in vendor data."
    else:
        if isinstance(required_value, str) and required_value.lower() in str(vendor_value).lower():
            result["status"] = "PASS"
            result["reasoning"] = f"Vendor value matches requirement."
        else:
            result["status"] = "REVIEW"
            result["reasoning"] = f"Could not auto-match. Vendor: '{vendor_value}', Required: '{required_value}'. Manual review recommended."
    
    return result


def generate_eligibility_report(results: list, vendor_profile: dict) -> dict:
    """
    Step 3: Generate overall eligibility summary from per-criterion results.
    """
    total = len(results)
    passed = sum(1 for r in results if r["status"] == "PASS")
    failed = sum(1 for r in results if r["status"] == "FAIL")
    review = sum(1 for r in results if r["status"] == "REVIEW")
    
    if failed == 0 and total > 0:
        overall = "ELIGIBLE"
        message = "✅ Vendor meets all extracted eligibility criteria."
    elif failed > 0:
        overall = "NOT ELIGIBLE"
        message = f"❌ Vendor fails {failed} out of {total} criteria."
    else:
        overall = "REVIEW NEEDED"
        message = "⚠️ Could not automatically determine eligibility. Manual review recommended."
    
    return {
        "overall_status": overall,
        "message": message,
        "total_criteria": total,
        "passed": passed,
        "failed": failed,
        "needs_review": review,
        "vendor_name": vendor_profile.get("company_name", "Unknown"),
        "details": results
    }
