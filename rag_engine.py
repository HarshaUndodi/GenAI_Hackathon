"""
RAG Engine Module.
Handles FAISS vector store creation and retrieval-augmented generation
for the tender document chatbot.
"""

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.chains import create_retrieval_chain
from langchain_core.prompts import ChatPromptTemplate
from prompts import RAG_SYSTEM_PROMPT


# Cache embeddings model globally to avoid re-downloading
_embeddings = None


def get_embeddings():
    """Get or create the embeddings model (cached globally)."""
    global _embeddings
    if _embeddings is None:
        _embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    return _embeddings


def build_vector_store(raw_text: str) -> FAISS:
    """
    Split text into chunks, embed with all-MiniLM-L6-v2, and index in FAISS.
    
    Chunking strategy: 1000 chars with 200 char overlap.
    This ensures context continuity across chunk boundaries.
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    
    chunks = text_splitter.split_text(raw_text)
    embeddings = get_embeddings()
    vector_store = FAISS.from_texts(chunks, embeddings)
    
    return vector_store


def get_rag_chain(vector_store: FAISS, llm: ChatGroq):
    """
    Create a LangChain RetrievalQA chain for querying the tender document.
    """
    retriever = vector_store.as_retriever(search_kwargs={"k": 5})
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", RAG_SYSTEM_PROMPT),
        ("human", "{input}")
    ])
    
    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    rag_chain = create_retrieval_chain(retriever, question_answer_chain)
    
    return rag_chain


def query_tender(vector_store: FAISS, llm: ChatGroq, question: str) -> str:
    """
    Query the tender document using RAG.
    Returns the LLM's answer grounded in retrieved context.
    """
    chain = get_rag_chain(vector_store, llm)
    
    try:
        response = chain.invoke({"input": question})
        return response.get("answer", "No answer generated.")
    except Exception as e:
        return f"Error querying tender: {str(e)}"
