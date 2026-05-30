import os
from dotenv import load_dotenv
load_dotenv()
from langchain_groq import ChatGroq
from pdf_processing import extract_text
from eligibility_engine import extract_criteria, check_eligibility

text = extract_text('data/1.pdf')
llm = ChatGroq(model='llama-3.1-8b-instant', temperature=0)
criteria = extract_criteria(text, llm)

vendor_profile = {
    "company_name": "Acme Innovations Ltd.",
    "turnover": 7500.0,
    "experience_years": 8,
    "projects_completed": 25,
    "msme_status": True,
    "iso_certified": True,
    "certifications": ["ISO 14001", "CMMI Level 3", "ISO 9001"]
}

results = check_eligibility(criteria, vendor_profile)

for r in results:
    if r["status"] == "FAIL":
        print(f"FAILED: {r['criterion']}")
        print(f"  Field: {r.get('field', 'Unknown')} | Required: {r['required']} | Vendor: {r['vendor_value']}")
        print(f"  Reason: {r['reasoning']}\n")
