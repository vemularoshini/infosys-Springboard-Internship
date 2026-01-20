import os
import streamlit as st
from dotenv import load_dotenv
import numpy as np
import re

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def get_hf_token():
    # Priority 1: Hugging Face Space secrets
    try:
        if "HF_TOKEN" in st.secrets:
            return st.secrets["HF_TOKEN"]
    except Exception:
        pass

    # Priority 2: Environment variable (Docker / local dev)
    return os.getenv("HF_TOKEN")

HF_TOKEN = get_hf_token()

if not HF_TOKEN:
    st.error("HuggingFace token not found. Please configure HF_TOKEN in Space Secrets.")
    st.stop()


# LangChain + tools
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import (
    HuggingFaceEmbeddings,
    HuggingFaceEndpoint,
    ChatHuggingFace,
)
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate

# -----------------------------------------------------------
# Load .env
# -----------------------------------------------------------
# load_dotenv()
# hf_token = st.secrets["HF_TOKEN"]

# -----------------------------------------------------------
# Page Setup
# -----------------------------------------------------------
st.set_page_config(
    page_title="RAG Multi‑PDF AI Assistant",
    layout="wide",
)

st.markdown("""
    <h1 style='text-align: center; color:#4A90E2;'>
         RAG‑Powered Multi‑PDF Question Answering Assistant
    </h1>
    <p style='text-align: center; font-size:17px;'>
        Ask any question related to the information stored inside the PDFs.<br>
        If the content exists, the AI will give you the exact answer!
    </p>
""", unsafe_allow_html=True)

if not HF_TOKEN:
    st.error(" HF_TOKEN missing in `.env`. Please add your HuggingFace token.")
    st.stop()

# -----------------------------------------------------------
# PDF Paths (both PDFs)
# -----------------------------------------------------------
PDF_PATHS = [
    os.path.join(BASE_DIR, "rag_sample_qa.pdf"),
    os.path.join(BASE_DIR, "knowledge_base.pdf"),
]

TOP_K = 4  # number of chunks retrieved
SIM_THRESHOLD = 0.38  # semantic gate to detect obviously irrelevant retrieval (tuned to avoid false negatives)

# -----------------------------------------------------------
# Utility helpers
# -----------------------------------------------------------
def normalize_space(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "")).strip()

FALLBACK_PHRASE = "i don't know based on the provided context."
def is_fallback_answer(text: str) -> bool:
    """
    Return True if the model's answer is essentially the fallback sentence.
    Normalize whitespace, quotes, punctuation, and case.
    """
    if not text:
        return True
    normalized = normalize_space(text).strip('"\'').lower().strip(" .!?\n\t")
    return normalized == FALLBACK_PHRASE or normalized.startswith(FALLBACK_PHRASE)

def max_cosine_similarity(embeddings, query: str, docs) -> float:
    """
    Compute max cosine similarity between query embedding and each retrieved doc chunk.
    """
    try:
        q = np.array(embeddings.embed_query(query), dtype=np.float32)
        cosines = []
        for d in docs:
            dv = np.array(embeddings.embed_documents([d.page_content])[0], dtype=np.float32)
            qn = q / (np.linalg.norm(q) + 1e-12)
            dn = dv / (np.linalg.norm(dv) + 1e-12)
            cosines.append(float((qn * dn).sum()))
        return max(cosines) if cosines else 0.0
    except Exception:
        return 0.0

def linkify_urls(text: str) -> str:
    """
    Convert plain URLs to clickable links. Works even if model returns raw text (not markdown links).
    """
    if not text:
        return ""
    # Replace http/https URLs with clickable <a> tags
    def repl(m):
        url = m.group(0)
        return f'{url}{url}</a>'
    return re.sub(r'(https?://[^\s\])>]+)', repl, text)

def render_sorry_panel():
    st.markdown("""
    <div style='padding:20px; background-color:#ffe6e6; border-radius:10px;'>
        <h3>Sorry!</h3>
        <p style='font-size:17px; margin-bottom:0;'>We couldn't find an answer to your question in the PDFs.</p>
        <p style='font-size:16px; margin-top:8px;'>
            Please try another question or rephrase your query (e.g., “How do I reset my password?”, “How to create an account?”, “How do I enable two‑factor authentication?”).
        </p>
    </div>
    """, unsafe_allow_html=True)

def render_answer_panel(answer_text_html: str):
    st.markdown("""
        <h3 style='color:#4CAF50;'> Answer Found</h3>
    """, unsafe_allow_html=True)
    st.markdown(
        f"""
        <div style='background-color:#e8f5e9; padding:15px; border-radius:10px; font-size:17px;'>
            {answer_text_html}
        </div>
        """,
        unsafe_allow_html=True,
    )

# -----------------------------------------------------------
# Build Vector DB (cached)
# -----------------------------------------------------------
@st.cache_resource
def build_db(pdf_paths):
    all_docs = []
    for p in pdf_paths:
        if not os.path.exists(p):
            raise FileNotFoundError(f"PDF not found: {p}")
        loader = PyPDFLoader(p)
        all_docs.extend(loader.load())

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
    )
    chunks = splitter.split_documents(all_docs)

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vectordb = FAISS.from_documents(chunks, embeddings)
    return vectordb, embeddings

try:
    vectordb, embeddings = build_db(PDF_PATHS)
    #st.success("Both PDFs loaded & indexed successfully!")
except Exception as e:
    st.error(f"Error building FAISS DB: {e}")
    st.stop()

# -----------------------------------------------------------
# LLM Setup
# -----------------------------------------------------------
llm_endpoint = HuggingFaceEndpoint(
    repo_id="openai/gpt-oss-20b",   # replace if access is restricted; e.g. "mistralai/Mistral-7B-Instruct-v0.2"
    huggingfacehub_api_token=HF_TOKEN,
    temperature=0.2,
    max_new_tokens=350,
)

llm = ChatHuggingFace(llm=llm_endpoint)

prompt_template = ChatPromptTemplate.from_template("""
You are an AI assistant that answers ONLY from the provided context.
If the answer is not found in the context, reply:
"I don't know based on the provided context."
Question:
{question}
Context:
{context}
Answer:
""")

# -----------------------------------------------------------
# UI - Ask Question
# -----------------------------------------------------------
st.markdown("<h3 style='color:#333;'>Ask Your Question Below </h3>", unsafe_allow_html=True)

question = st.text_area(
    "Type your question:",
    height=130,
    placeholder="Example: How do I reset my password?  •  How to create an account?  •  How do I change billing details?",
)

submit_btn = st.button("Submit")

if submit_btn:

    if not question or not question.strip():
        st.warning("Please type a valid question.")
        st.stop()

    # Retrieve chunks
    retriever = vectordb.as_retriever(search_kwargs={"k": TOP_K})
    retrieved_docs = retriever.invoke(question)

    # If FAISS returns nothing → Sorry panel, stop.
    if len(retrieved_docs) == 0:
        render_sorry_panel()
        st.stop()

    # Semantic check (to avoid misleading answers if everything is irrelevant)
    sim_score = max_cosine_similarity(embeddings, question, retrieved_docs)

    # Prepare context for LLM
    context_text = "\n\n".join([d.page_content for d in retrieved_docs])

    # If similarity is too low → Sorry panel (don’t call LLM)
    if sim_score < SIM_THRESHOLD:
        render_sorry_panel()
        st.stop()

    # Call LLM with context
    chain = prompt_template | llm
    try:
        result = chain.invoke({"question": question, "context": context_text})
        answer_text = result.content if hasattr(result, "content") else str(result)
    except Exception as e:
        st.error(f"LLM error: {e}")
        st.stop()

    # If the model returned the fallback sentence → Sorry panel
    if is_fallback_answer(answer_text):
        render_sorry_panel()
    else:
        # Make URLs clickable
        answer_html = linkify_urls(answer_text)
        render_answer_panel(answer_html)

else:
    st.info("Tip: Ask a question above and click Submit!")

# -----------------------------------------------------------
# Footer
# -----------------------------------------------------------
st.markdown("""
    <br><hr>
    <p style='text-align:center; font-size:14px; color:gray;'>
        Built with using Streamlit, FAISS, HuggingFace & LangChain.
    </p>
""", unsafe_allow_html=True)