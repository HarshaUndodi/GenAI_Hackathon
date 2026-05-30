# 🚀 GenAI Hackathon: Tender Simplifier

**Tender Simplifier** is an AI-powered SaaS tool designed to help Small and Medium Enterprises (SMEs) navigate the incredibly complex world of Government Tenders. It leverages state-of-the-art Large Language Models (LLMs) via **Groq** to break down massive, jargon-filled PDF tender documents into plain language, auto-evaluating vendor eligibility in seconds.

---

## ✨ Features

- **📄 AI-Generated Plain Language Summary**: Upload a 50+ page tender PDF and instantly receive a human-readable summary of the scope, deadlines, financial requirements, and required documents.
- **🤖 Hybrid Eligibility Engine**: 
  - *Extraction*: The LLM meticulously extracts numeric, boolean, and text-based eligibility criteria.
  - *Evaluation*: A deterministic Python engine evaluates your company profile against these criteria, eliminating LLM hallucinations in the decision-making process.
- **📅 Timeline Tracker**: Automatically extracts critical dates (Submission Deadline, Pre-bid meeting, etc.) and presents them chronologically.
- **💬 TenderBot RAG Chatbot**: Ask highly specific questions about the tender document and get cited answers instantly using Retrieval-Augmented Generation (RAG).
- **📦 Export Bid Package**: Instantly download a ready-to-go `.zip` file containing a Bid Checklist and a detailed Eligibility Report to share with your team.
- **🎨 Premium Dark Mode UI**: A highly polished, sleek UI designed with Streamlit that feels like a professional B2B SaaS platform.

---

## 🛠️ Architecture

- **Frontend**: [Streamlit](https://streamlit.io/) with custom CSS theming
- **LLM Engine**: [LangChain](https://www.langchain.com/) + [Groq](https://groq.com/) API (Llama 3.3 70B / Llama 3.1 8B)
- **RAG Stack**: FAISS (Vector Store), HuggingFace Embeddings (`all-MiniLM-L6-v2`)
- **Document Processing**: `pdfplumber` for table/text extraction, `python-docx` for report generation

---

## 🚀 Quick Start Setup

### 1. Clone the repository
```bash
git clone https://github.com/HarshaUndodi/GenAI_Hackathon.git
cd GenAI_Hackathon
```

### 2. Create a Virtual Environment and Install Dependencies
```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Set up Environment Variables
Rename `.env.example` to `.env` (or create a new `.env` file) and add your Groq API Key:
```env
GROQ_API_KEY=gsk_your_groq_api_key_here
```
*Note: You can get a free API key from the [Groq Console](https://console.groq.com/).*

### 4. Run the Application
```bash
python -m streamlit run app.py
```
Open your browser to `http://localhost:8501` and upload a tender PDF!

---

## 💡 Built for the GenAI Hackathon
This project was built to solve the real-world problem of SMEs spending weeks analyzing government tenders only to find out they are ineligible. **Tender Simplifier** reduces that time from weeks to seconds.
