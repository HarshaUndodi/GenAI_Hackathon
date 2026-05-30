"""
Government Tender Simplifier & Bid Eligibility Checker
Main Streamlit Application - Phase 2 Revamp
"""

from dotenv import load_dotenv
load_dotenv()

import streamlit as st
import os
import json
import requests
from langchain_groq import ChatGroq
from streamlit_lottie import st_lottie

from pdf_processing import extract_text, get_page_count
from section_extractor import extract_sections, generate_summary
from eligibility_engine import extract_criteria, check_eligibility, generate_eligibility_report
from rag_engine import build_vector_store, query_tender
from doc_generator import generate_checklist_docx, generate_eligibility_report_docx, generate_zip_package
from prompts import TIMELINE_EXTRACTION_PROMPT, CHECKLIST_PROMPT

# ──────────────────────────────────────────────
# PAGE CONFIG
# ──────────────────────────────────────────────
st.set_page_config(
    page_title="GovTender AI — Premium Edition",
    page_icon="🏛️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ──────────────────────────────────────────────
# LOTTIE ANIMATIONS
# ──────────────────────────────────────────────
@st.cache_data
def load_lottieurl(url: str):
    try:
        r = requests.get(url)
        if r.status_code != 200:
            return None
        return r.json()
    except Exception:
        return None

lottie_loading = load_lottieurl("https://lottie.host/80eb6d03-e62a-436d-bef6-d748f22e8ec5/R9gT3WlXm8.json")

# ──────────────────────────────────────────────
# CUSTOM STYLING (Augmenting config.toml)
# ──────────────────────────────────────────────
st.markdown("""
<style>
    /* Global App Background overriding */
    .stApp {
        background-color: #0F172A;
        color: #F8FAFC;
    }
    
    /* Main Header Styling */
    .main-header {
        text-align: center;
        padding: 3rem 0;
        background: linear-gradient(135deg, #1E3A8A 0%, #0F172A 100%);
        border-radius: 16px;
        margin-bottom: 2.5rem;
        color: white;
        border: 1px solid #334155;
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.5);
    }
    .main-header h1 {
        color: #F8FAFC !important;
        font-size: 3rem !important;
        margin: 0 !important;
        font-weight: 800;
        letter-spacing: -1px;
    }
    .main-header p {
        color: #94A3B8;
        margin: 0.75rem 0 0 0;
        font-size: 1.2rem;
        font-weight: 400;
    }
    
    /* Status Cards */
    .status-pass {
        background: rgba(16, 185, 129, 0.1);
        border-left: 4px solid #10B981;
        padding: 16px 20px;
        border-radius: 8px;
        margin: 12px 0;
        border-top: 1px solid rgba(16, 185, 129, 0.2);
        border-right: 1px solid rgba(16, 185, 129, 0.2);
        border-bottom: 1px solid rgba(16, 185, 129, 0.2);
    }
    .status-fail {
        background: rgba(239, 68, 68, 0.1);
        border-left: 4px solid #EF4444;
        padding: 16px 20px;
        border-radius: 8px;
        margin: 12px 0;
        border-top: 1px solid rgba(239, 68, 68, 0.2);
        border-right: 1px solid rgba(239, 68, 68, 0.2);
        border-bottom: 1px solid rgba(239, 68, 68, 0.2);
    }
    .status-review {
        background: rgba(245, 158, 11, 0.1);
        border-left: 4px solid #F59E0B;
        padding: 16px 20px;
        border-radius: 8px;
        margin: 12px 0;
        border-top: 1px solid rgba(245, 158, 11, 0.2);
        border-right: 1px solid rgba(245, 158, 11, 0.2);
        border-bottom: 1px solid rgba(245, 158, 11, 0.2);
    }
    
    /* KPI Cards Top Level */
    .kpi-card {
        background: #1E293B;
        padding: 24px;
        border-radius: 12px;
        text-align: center;
        border: 1px solid #334155;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.3);
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    .kpi-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.4);
    }
    .kpi-card h3 {
        margin: 0;
        color: #38BDF8;
        font-size: 2.2rem;
        font-weight: 700;
    }
    .kpi-card p {
        margin: 6px 0 0 0;
        color: #94A3B8;
        font-size: 0.85rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
</style>
""", unsafe_allow_html=True)

# ──────────────────────────────────────────────
# HEADER
# ──────────────────────────────────────────────
st.markdown("""
<div class="main-header">
    <h1>🏛️ GovTender AI</h1>
    <p>Premium Tender Document Simplifier & Hybrid Bid Eligibility Engine</p>
</div>
""", unsafe_allow_html=True)

# ──────────────────────────────────────────────
# SESSION STATE INITIALIZATION
# ──────────────────────────────────────────────
st.session_state.llm = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0,
    max_tokens=2000
)
for key in ["raw_text", "vector_store", "sections", "criteria", "eligibility_report", "tender_name", "page_count"]:
    if key not in st.session_state:
        st.session_state[key] = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "demo_profile" not in st.session_state:
    st.session_state.demo_profile = None

# ──────────────────────────────────────────────
# SIDEBAR — DOCUMENT UPLOAD & SETTINGS
# ──────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 📄 Upload Tender Document")
    uploaded_file = st.file_uploader(
        "Upload a government tender PDF",
        type=["pdf"],
        help="Upload the tender document you want to analyze"
    )
    st.markdown("---")
    st.markdown("### ⚙️ Demo Controls")
    
    if st.button("🟢 Load Dummy Profile (Eligible)", use_container_width=True):
        st.session_state.demo_profile = {
            "name": "Acme Innovations Ltd.",
            "turnover": 7500.0,
            "experience": 8,
            "projects": 25,
            "msme": "Medium Enterprise",
            "iso": "Certified",
            "certs": "ISO 14001, CMMI Level 3"
        }
        st.toast("✅ Loaded Eligible Profile!")
        st.rerun()
        
    if st.button("🔴 Load Dummy Profile (Ineligible)", use_container_width=True):
        st.session_state.demo_profile = {
            "name": "Startup XYZ",
            "turnover": 50.0,
            "experience": 1,
            "projects": 2,
            "msme": "Micro Enterprise",
            "iso": "Not Certified",
            "certs": ""
        }
        st.toast("❌ Loaded Ineligible Profile!")
        st.rerun()
        
    if st.button("🧹 Clear Profile", use_container_width=True):
        st.session_state.demo_profile = None
        st.rerun()

# ──────────────────────────────────────────────
# MAIN UI FLOW
# ──────────────────────────────────────────────

if uploaded_file:
    # Process File
    if st.session_state.tender_name != uploaded_file.name:
        st.session_state.tender_name = uploaded_file.name
        st.session_state.raw_text = None
        st.session_state.vector_store = None
        st.session_state.sections = None
        
    os.makedirs("data", exist_ok=True)
    temp_path = os.path.join("data", uploaded_file.name)
    with open(temp_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    if st.session_state.raw_text is None:
        with st.spinner("📖 Extracting text and parsing tables..."):
            st.session_state.raw_text = extract_text(temp_path)
            st.session_state.page_count = get_page_count(temp_path)
        
        if not st.session_state.raw_text:
            st.error("❌ Could not extract text from this PDF. It might be scanned/image-based.")
            st.stop()
    
    if st.session_state.vector_store is None:
        with st.spinner("🔍 Building FAISS vector index for semantic AI search..."):
            st.session_state.vector_store = build_vector_store(st.session_state.raw_text)
        st.toast("✅ AI Engine Ready!", icon="🚀")

    # Top KPI Dashboard
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(f'<div class="kpi-card"><p>Pages Analysed</p><h3>{st.session_state.page_count}</h3></div>', unsafe_allow_html=True)
    with col2:
        st.markdown(f'<div class="kpi-card"><p>Words Indexed</p><h3>{len(st.session_state.raw_text.split()):,}</h3></div>', unsafe_allow_html=True)
    with col3:
        status_text = "Ready" if st.session_state.vector_store else "Processing"
        st.markdown(f'<div class="kpi-card"><p>AI Engine Status</p><h3>{status_text}</h3></div>', unsafe_allow_html=True)
    with col4:
        st.markdown(f'<div class="kpi-card"><p>Model Active</p><h3>Llama 70B</h3></div>', unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)

    # Main Tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📋 AI Summary",
        "🏢 Vendor Profile & Eligibility",
        "📅 Timeline Tracker",
        "📝 Export Bid Package",
        "💬 TenderBot Chat"
    ])
    
    # ════════════════════════════════════════════
    # TAB 1: SUMMARY
    # ════════════════════════════════════════════
    with tab1:
        st.markdown("### 📋 AI-Generated Plain Language Summary")
        if not st.session_state.sections:
            if st.button("🔍 Generate Breakdown", type="primary"):
                with st.spinner("Analyzing document structure..."):
                    if lottie_loading:
                        st_lottie(lottie_loading, height=150, key="loading_sum")
                    st.session_state.sections = extract_sections(st.session_state.raw_text, st.session_state.llm)
                    st.session_state.summary = generate_summary(st.session_state.raw_text, st.session_state.llm)
                st.rerun()
        else:
            st.markdown(st.session_state.summary)
            
            st.markdown("---")
            st.markdown("### 📂 Extracted Raw Sections")
            for title, key in [("Scope of Work", "scope"), ("Eligibility", "eligibility"), ("EMD/Financials", "emd"), ("Key Dates", "dates"), ("Required Documents", "documents")]:
                with st.expander(title):
                    content = st.session_state.sections.get(key, "")
                    if isinstance(content, list):
                        for c in content: st.markdown(f"- {c}")
                    else:
                        st.markdown(content)
                        
    # ════════════════════════════════════════════
    # TAB 2: VENDOR & ELIGIBILITY
    # ════════════════════════════════════════════
    with tab2:
        st.markdown("### 🏢 Step 1: Define Vendor Profile")
        
        # Load demo profile if exists
        dp = st.session_state.demo_profile or {}
        
        with st.form("vendor_form"):
            c1, c2 = st.columns(2)
            with c1:
                v_name = st.text_input("Company Name", value=dp.get("name", ""))
                v_turnover = st.number_input("Annual Turnover (₹ Lakhs)", value=dp.get("turnover", 0.0), step=10.0)
                v_msme = st.selectbox("MSME Status", ["Not Registered", "Micro Enterprise", "Small Enterprise", "Medium Enterprise"], 
                                      index=["Not Registered", "Micro Enterprise", "Small Enterprise", "Medium Enterprise"].index(dp.get("msme", "Not Registered")))
            with c2:
                v_exp = st.number_input("Experience (Years)", value=int(dp.get("experience", 0)), step=1)
                v_proj = st.number_input("Projects Completed", value=int(dp.get("projects", 0)), step=1)
                v_iso = st.selectbox("ISO 9001 Certification", ["Not Certified", "Certified"], 
                                     index=["Not Certified", "Certified"].index(dp.get("iso", "Not Certified")))
            
            v_certs = st.text_input("Other Certifications (comma-separated)", value=dp.get("certs", ""))
            
            submit_profile = st.form_submit_button("💾 Save Profile & Run Eligibility Check", type="primary")

        if submit_profile:
            vendor_profile = {
                "company_name": v_name, "turnover": v_turnover, "experience_years": v_exp,
                "projects_completed": v_proj, "msme_status": v_msme != "Not Registered",
                "iso_certified": v_iso == "Certified", 
                "certifications": [c.strip() for c in v_certs.split(",") if c.strip()] if v_certs else []
            }
            if v_iso == "Certified" and "ISO 9001" not in vendor_profile["certifications"]:
                vendor_profile["certifications"].append("ISO 9001")
            
            st.session_state.current_vendor = vendor_profile

            with st.spinner("🤖 AI extracting criteria & Python Engine evaluating..."):
                if lottie_loading: st_lottie(lottie_loading, height=150, key="loading_el")
                criteria = extract_criteria(st.session_state.raw_text, st.session_state.llm)
                if criteria:
                    results = check_eligibility(criteria, vendor_profile)
                    st.session_state.eligibility_report = generate_eligibility_report(results, vendor_profile)
                    st.session_state.criteria = criteria
                else:
                    st.error("Could not extract criteria from this document.")

        if st.session_state.eligibility_report:
            st.markdown("---")
            st.markdown("### ✅ Step 2: Hybrid Evaluation Results")
            report = st.session_state.eligibility_report
            
            if report["overall_status"] == "ELIGIBLE":
                st.success(f"## {report['message']}")
            elif report["overall_status"] == "NOT ELIGIBLE":
                st.error(f"## {report['message']}")
            else:
                st.warning(f"## {report['message']}")
                
            for item in report["details"]:
                css_class = "status-pass" if item["status"] == "PASS" else "status-fail" if item["status"] == "FAIL" else "status-review"
                icon = "✅" if item["status"] == "PASS" else "❌" if item["status"] == "FAIL" else "⚠️"
                st.markdown(f"""
                <div class="{css_class}">
                    <strong>{icon} {item['criterion']}</strong><br>
                    <small>Required: {item['required']} | Your Value: {item['vendor_value']} | <strong>{item['status']}</strong></small><br>
                    <small>💬 {item['reasoning']}</small><br>
                    <small style="color:#666;"><em>📌 Source: "{item.get('source_text', 'N/A')}"</em></small>
                </div>
                """, unsafe_allow_html=True)
                
    # ════════════════════════════════════════════
    # TAB 3: TIMELINE
    # ════════════════════════════════════════════
    with tab3:
        st.markdown("### 📅 Extracted Timeline & Deadlines")
        if st.button("Extract Dates", type="primary"):
            with st.spinner("Mapping chronologies..."):
                response = st.session_state.llm.invoke(TIMELINE_EXTRACTION_PROMPT.format(text=st.session_state.raw_text[:12000]))
                try:
                    content = response.content.strip()
                    if content.startswith("```"): content = content.split("\n", 1)[1].rsplit("\n", 1)[0].strip()
                    timeline = json.loads(content)
                    
                    for i, event in enumerate(timeline):
                        icon = "🔴" if event.get("priority") == "critical" else "🟡" if event.get("priority") == "important" else "🔵"
                        c1, c2 = st.columns([1,3])
                        with c1:
                            st.markdown(f"### {icon} {event.get('date', 'N/A')}")
                            st.caption(event.get('time', ''))
                        with c2:
                            st.markdown(f"**{event.get('event', 'N/A')}**")
                            st.caption(event.get('description', ''))
                        st.markdown("---")
                except:
                    st.error("Failed to parse timeline.")
                    st.write(response.content)

    # ════════════════════════════════════════════
    # TAB 4: EXPORT PACKAGE
    # ════════════════════════════════════════════
    with tab4:
        st.markdown("### 📦 Export Complete Bid Package")
        st.caption("Generate a ZIP file containing your compliance checklist and eligibility report.")
        
        if not st.session_state.eligibility_report:
            st.warning("⚠️ Please run the Eligibility Check (Tab 2) first to generate the report.")
        else:
            if st.button("Generate .ZIP Package", type="primary", use_container_width=True):
                with st.spinner("Compiling documents..."):
                    # Generate Checklist
                    checklist_output = query_tender(st.session_state.vector_store, st.session_state.llm, CHECKLIST_PROMPT.format(text="Generate bid checklist"))
                    checklist_io = generate_checklist_docx(uploaded_file.name, checklist_output)
                    
                    # Generate Report
                    report_io = generate_eligibility_report_docx(st.session_state.eligibility_report, st.session_state.current_vendor, uploaded_file.name)
                    
                    # Zip it
                    zip_io = generate_zip_package(uploaded_file.name, checklist_io, report_io)
                    
                    st.download_button(
                        label="📥 Download Full Bid Package (.ZIP)",
                        data=zip_io,
                        file_name=f"GovTender_Package_{uploaded_file.name.replace('.pdf', '')}.zip",
                        mime="application/zip",
                        type="primary",
                        use_container_width=True
                    )
                    st.success("✅ Package is ready for download!")

    # ════════════════════════════════════════════
    # TAB 5: CHATBOT
    # ════════════════════════════════════════════
    with tab5:
        st.markdown("### 💬 Ask TenderBot")
        
        for msg in st.session_state.chat_history:
            avatar = "🏢" if msg["role"] == "user" else "🤖"
            with st.chat_message(msg["role"], avatar=avatar):
                st.markdown(msg["content"])
                
        user_question = st.chat_input("E.g., What is the bank guarantee required?")
        
        if user_question:
            st.session_state.chat_history.append({"role": "user", "content": user_question})
            with st.chat_message("user", avatar="🏢"):
                st.markdown(user_question)
                
            with st.chat_message("assistant", avatar="🤖"):
                with st.spinner("Searching..."):
                    answer = query_tender(st.session_state.vector_store, st.session_state.llm, user_question)
                st.markdown(answer)
                st.session_state.chat_history.append({"role": "assistant", "content": answer})

else:
    st.markdown("""
    <div style="text-align: center; color: #5f6368; padding: 4rem 2rem;">
        <h2 style="color: #004fb0;">Simplify Gov Tenders in 3 Steps</h2>
        <br>
        <div style="display: flex; justify-content: center; gap: 2rem;">
            <div class="kpi-card" style="flex:1;">
                <h1 style="font-size: 3rem;">1</h1>
                <p>Upload PDF</p>
            </div>
            <div class="kpi-card" style="flex:1;">
                <h1 style="font-size: 3rem;">2</h1>
                <p>Run Eligibility</p>
            </div>
            <div class="kpi-card" style="flex:1;">
                <h1 style="font-size: 3rem;">3</h1>
                <p>Export Package</p>
            </div>
        </div>
        <br><br>
        <p><i>Awaiting PDF upload in the sidebar...</i></p>
    </div>
    """, unsafe_allow_html=True)