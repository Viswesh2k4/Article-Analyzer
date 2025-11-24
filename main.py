import os
import time
import pickle
import requests
import streamlit as st

from bs4 import BeautifulSoup
from dotenv import load_dotenv

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq

# Optional: newspaper3k for extra robustness
try:
    from newspaper import Article
except Exception:
    Article = None


# ---------- ENV + API KEY ----------
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

st.set_page_config(page_title="Article Reader", page_icon="📰")

st.title("📰 Article Analyzer")
st.sidebar.title("Article URLs")

if not GROQ_API_KEY:
    st.error("GROQ_API_KEY not found in environment. Set it in .env to query the LLM.")
    st.stop()

# Groq LLM (using native Groq client via LangChain)
llm = ChatGroq(
    api_key=GROQ_API_KEY,
    model="llama-3.1-8b-instant",
    temperature=0.2,
    max_tokens=500,
)

# ---------- URL INPUT ----------
urls = []
for i in range(3):
    url = st.sidebar.text_input(f"URL {i + 1}")
    if url:
        urls.append(url)

process_url_clicked = st.sidebar.button("Process URLs")
file_path = "faiss_store_groq_articles.pkl"

main_placeholder = st.empty()


# ---------- Helper: Fetch article text ----------
def fetch_url_text(url: str) -> str:
    """Fetch main text from a news/article URL using requests + BS4, fallback to newspaper3k."""
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/129.0 Safari/537.36"
        )
    }

    # 1) Try requests + BeautifulSoup
    try:
        resp = requests.get(url, headers=headers, timeout=20)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")

        # Remove scripts/styles
        for tag in soup(["script", "style", "noscript"]):
            tag.decompose()

        text = soup.get_text(" ", strip=True)
        if text and len(text) > 500:
            return text
    except Exception:
        pass

    # 2) Fallback to newspaper3k if available
    if Article is not None:
        try:
            art = Article(url)
            art.download()
            art.parse()
            txt = (art.title or "") + "\n\n" + (art.text or "")
            if txt and len(txt) > 300:
                return txt
        except Exception:
            pass

    return ""


# ---------- Process URLs & Build Vector Store ----------
if process_url_clicked:
    main_placeholder.text("Data Loading... Started... ✅")
    docs_loaded = []
    st.session_state["processed_urls"] = []

    for u in urls:
        if not u:
            continue
        content = fetch_url_text(u)
        if content:
            docs_loaded.append(Document(page_content=content, metadata={"source": u}))
            st.session_state["processed_urls"].append(u)

    if not docs_loaded:
        main_placeholder.text(
            "No text could be fetched from the provided URLs. "
            "Try different links or paste full article URLs."
        )
        st.stop()

    # Split text into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", ".", ","],
        chunk_size=1000,
        chunk_overlap=150,
    )
    main_placeholder.text("Text Splitter... Started... ✂️")
    docs = text_splitter.split_documents(docs_loaded)

    if not docs:
        main_placeholder.text("No chunks were created from the articles.")
        st.stop()

    # Embeddings + FAISS
    main_placeholder.text("Embedding Vector Started Building... ⚙️")
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    vectorstore = FAISS.from_documents(docs, embeddings)

    # Save index
    with open(file_path, "wb") as f:
        pickle.dump(vectorstore, f)

    time.sleep(1)
    main_placeholder.text(f"FAISS index built and saved ✅ (chunks: {len(docs)})")


# ---------- Question / Answer (Manual RAG) ----------
st.subheader("Ask a question about the articles")
query = main_placeholder.text_input("Question:")

if query:
    if not os.path.exists(file_path):
        st.error("No FAISS index found. Please enter URLs and click 'Process URLs' first.")
    else:
        with open(file_path, "rb") as f:
            vectorstore = pickle.load(f)

        # Retrieve similar chunks
        docs = vectorstore.similarity_search(query, k=4)

        if not docs:
            st.error("No relevant chunks found. Try another question or different URLs.")
        else:
            # Show retrieved context for debugging
            with st.expander("🔍 Retrieved context (debug view)"):
                for i, d in enumerate(docs, start=1):
                    st.markdown(f"**Chunk {i}:**")
                    st.write(d.page_content[:1000])
                    src = d.metadata.get("source", "unknown")
                    st.caption(f"Source: {src}")
                    st.markdown("---")

            # Build context for the LLM
            context_texts = []
            sources = []
            for i, doc in enumerate(docs, start=1):
                context_texts.append(doc.page_content)
                source = doc.metadata.get("source", f"doc_{i}")
                sources.append(source)

            context = "\n\n---\n\n".join(context_texts)
            unique_sources = sorted({s for s in sources if s})

            prompt = f"""
You are a careful assistant that answers questions based ONLY on the provided article context.

Context:
{context}

Question: {query}

Rules:
- Use ONLY the information in the Context.
- If the answer is not clearly in the context, say: "I'm not sure based on the given articles."
- Be concise and factual.
            """.strip()

            with st.spinner("Thinking..."):
                response = llm.invoke(prompt)

            st.header("Answer")
            st.write(response.content if hasattr(response, "content") else str(response))

            if not unique_sources:
                unique_sources = st.session_state.get("processed_urls", [])

            if unique_sources:
                st.subheader("Sources:")
                for src in unique_sources:
                    st.write(src)
