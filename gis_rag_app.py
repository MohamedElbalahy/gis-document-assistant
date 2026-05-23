"""
🗺️ GIS Document Assistant - RAG-Powered PDF Chat
ITI GIS Track | Gen AI Course - Day 4 Lab Project

Features:
✅ Upload multiple GIS PDFs
✅ Chat with your documents (RAG)
✅ Source citations
✅ Export chat to JSON
✅ Arabic language support
✅ Usage statistics
"""

import os
import json
import tempfile
import datetime
import streamlit as st

from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader


# ─────────────────────────────────────────────
# Page Config
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="GIS Document Assistant",
    page_icon="🗺️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
# Custom CSS — dark GIS-themed UI
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans+Arabic:wght@300;400;600&display=swap');

html, body, [class*="css"] {
    font-family: 'IBM Plex Sans Arabic', 'IBM Plex Mono', sans-serif;
}

.stApp {
    background: #0d1117;
    color: #c9d1d9;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background: #161b22;
    border-right: 1px solid #30363d;
}

/* Title banner */
.hero-banner {
    background: linear-gradient(135deg, #1a3a5c 0%, #0d2137 50%, #0a1628 100%);
    border: 1px solid #1f6feb;
    border-radius: 12px;
    padding: 24px 32px;
    margin-bottom: 20px;
    position: relative;
    overflow: hidden;
}
.hero-banner::before {
    content: '';
    position: absolute;
    top: -50%;
    left: -50%;
    width: 200%;
    height: 200%;
    background: radial-gradient(circle at 30% 50%, rgba(31,111,235,0.08) 0%, transparent 60%);
    pointer-events: none;
}
.hero-title {
    font-size: 28px;
    font-weight: 600;
    color: #58a6ff;
    margin: 0;
    letter-spacing: -0.5px;
}
.hero-sub {
    font-size: 13px;
    color: #8b949e;
    margin: 6px 0 0;
}

/* Stat cards */
.stats-row {
    display: flex;
    gap: 12px;
    margin: 12px 0 20px;
}
.stat-card {
    flex: 1;
    background: #161b22;
    border: 1px solid #30363d;
    border-radius: 10px;
    padding: 14px 18px;
    text-align: center;
}
.stat-number {
    font-size: 26px;
    font-weight: 600;
    color: #58a6ff;
    font-family: 'IBM Plex Mono', monospace;
}
.stat-label {
    font-size: 11px;
    color: #8b949e;
    text-transform: uppercase;
    letter-spacing: 1px;
    margin-top: 4px;
}

/* Chat messages */
.msg-user {
    background: #1c2b3a;
    border: 1px solid #1f6feb40;
    border-radius: 12px 12px 4px 12px;
    padding: 14px 18px;
    margin: 8px 0;
    color: #c9d1d9;
    max-width: 85%;
    margin-left: auto;
}
.msg-assistant {
    background: #161b22;
    border: 1px solid #30363d;
    border-radius: 12px 12px 12px 4px;
    padding: 14px 18px;
    margin: 8px 0;
    color: #c9d1d9;
    max-width: 85%;
}
.msg-role {
    font-size: 11px;
    color: #8b949e;
    text-transform: uppercase;
    letter-spacing: 1px;
    margin-bottom: 6px;
    font-family: 'IBM Plex Mono', monospace;
}

/* Source pill */
.source-pill {
    display: inline-block;
    background: #1a3a5c;
    border: 1px solid #1f6feb60;
    border-radius: 20px;
    padding: 3px 10px;
    font-size: 11px;
    color: #58a6ff;
    margin: 2px;
    font-family: 'IBM Plex Mono', monospace;
}

/* File badge */
.file-badge {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    background: #1a3a5c20;
    border: 1px solid #1f6feb40;
    border-radius: 8px;
    padding: 6px 12px;
    font-size: 12px;
    color: #79c0ff;
    margin: 4px 2px;
}

/* Button overrides */
.stButton > button {
    background: #1f6feb !important;
    color: white !important;
    border: none !important;
    border-radius: 8px !important;
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 13px !important;
    padding: 8px 20px !important;
    transition: all 0.2s !important;
}
.stButton > button:hover {
    background: #388bfd !important;
    transform: translateY(-1px);
    box-shadow: 0 4px 12px rgba(31,111,235,0.3) !important;
}

/* Inputs */
.stTextInput > div > div > input,
.stTextArea > div > div > textarea {
    background: #0d1117 !important;
    border: 1px solid #30363d !important;
    border-radius: 8px !important;
    color: #c9d1d9 !important;
    font-family: 'IBM Plex Mono', monospace !important;
}

/* Progress */
.stProgress > div > div {
    background: #1f6feb !important;
}

.section-header {
    font-size: 13px;
    font-weight: 600;
    color: #8b949e;
    text-transform: uppercase;
    letter-spacing: 1.5px;
    padding: 8px 0 6px;
    border-bottom: 1px solid #21262d;
    margin-bottom: 10px;
    font-family: 'IBM Plex Mono', monospace;
}

.arabic-note {
    direction: rtl;
    text-align: right;
    font-family: 'IBM Plex Sans Arabic', sans-serif;
    background: #161b22;
    border-right: 3px solid #1f6feb;
    border-radius: 4px;
    padding: 10px 14px;
    color: #8b949e;
    font-size: 13px;
}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# Session State Initialization
# ─────────────────────────────────────────────
defaults = {
    "messages": [],          # chat history: [{role, content, sources, timestamp}]
    "vectorstore": None,     # ChromaDB
    "processed_files": [],   # list of processed PDF names
    "total_chunks": 0,       # total chunks stored
    "stats": {
        "questions_asked": 0,
        "pages_read": 0,
        "pdfs_loaded": 0,
    },
    "language": "English",   # UI/answer language
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v


# ─────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown('<div class="section-header">⚙️ Configuration</div>', unsafe_allow_html=True)

    api_key = st.text_input(
        "Gemini API Key",
        type="password",
        placeholder="AIza...",
        help="Get yours at https://aistudio.google.com/apikey",
    )

    st.markdown("")
    st.markdown('<div class="section-header">🌐 Language / اللغة</div>', unsafe_allow_html=True)
    language = st.selectbox(
        "Answer language",
        ["English", "Arabic (العربية)", "Bilingual (EN + AR)"],
        index=0,
    )
    st.session_state.language = language

    st.markdown("")
    st.markdown('<div class="section-header">📄 Upload PDFs</div>', unsafe_allow_html=True)

    uploaded_files = st.file_uploader(
        "Upload GIS PDFs",
        type="pdf",
        accept_multiple_files=True,
        label_visibility="collapsed",
    )

    chunk_size = st.slider("Chunk size (chars)", 200, 1000, 500, 50)
    chunk_overlap = st.slider("Chunk overlap", 20, 200, 50, 10)

    process_btn = st.button("⚡ Process PDFs", use_container_width=True)

    st.markdown("")
    st.markdown('<div class="section-header">📊 Session Stats</div>', unsafe_allow_html=True)
    s = st.session_state.stats
    st.markdown(f"""
    <div style="font-family:'IBM Plex Mono',monospace; font-size:12px; color:#8b949e; line-height:2">
    📁 PDFs loaded: <span style="color:#58a6ff">{s['pdfs_loaded']}</span><br>
    🧩 Chunks stored: <span style="color:#58a6ff">{st.session_state.total_chunks}</span><br>
    ❓ Questions asked: <span style="color:#58a6ff">{s['questions_asked']}</span><br>
    📑 Pages sourced: <span style="color:#58a6ff">{s['pages_read']}</span>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("")
    st.markdown('<div class="section-header">💾 Export</div>', unsafe_allow_html=True)

    if st.button("⬇️ Export Chat (JSON)", use_container_width=True):
        if st.session_state.messages:
            export_data = {
                "exported_at": datetime.datetime.now().isoformat(),
                "pdfs": st.session_state.processed_files,
                "stats": st.session_state.stats,
                "conversation": [
                    {
                        "role": m["role"],
                        "content": m["content"],
                        "timestamp": m.get("timestamp", ""),
                        "sources": [
                            {
                                "source": s.metadata.get("source", "?"),
                                "page": s.metadata.get("page", "?"),
                                "snippet": s.page_content[:200],
                            }
                            for s in m.get("sources", [])
                        ],
                    }
                    for m in st.session_state.messages
                ],
            }
            json_str = json.dumps(export_data, ensure_ascii=False, indent=2)
            st.download_button(
                label="📥 Download JSON",
                data=json_str,
                file_name=f"gis_chat_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json",
                use_container_width=True,
            )
        else:
            st.warning("No messages to export yet.")

    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()


# ─────────────────────────────────────────────
# Main Area
# ─────────────────────────────────────────────

# Hero banner
st.markdown("""
<div class="hero-banner">
    <p class="hero-title">🗺️ GIS Document Assistant</p>
    <p class="hero-sub">RAG-powered chat · ITI GIS Track · Gen AI Day 4 Lab</p>
</div>
""", unsafe_allow_html=True)

# Guard: API key
if not api_key:
    st.markdown("""
    <div style="background:#1a2332;border:1px solid #f0883e40;border-radius:10px;padding:20px;text-align:center;color:#f0883e">
        ⚠️ Enter your <b>Gemini API Key</b> in the sidebar to get started.<br>
        <small style="color:#8b949e">Get one free at <a href="https://aistudio.google.com/apikey" target="_blank" style="color:#58a6ff">aistudio.google.com</a></small>
    </div>
    """, unsafe_allow_html=True)
    st.stop()

os.environ["GOOGLE_API_KEY"] = api_key


# ─────────────────────────────────────────────
# PDF Processing
# ─────────────────────────────────────────────
def build_language_instruction(lang: str) -> str:
    if lang == "Arabic (العربية)":
        return "أجب باللغة العربية فقط. اكتب إجابة واضحة ومفصلة بالعربية."
    elif lang == "Bilingual (EN + AR)":
        return (
            "Provide the answer in both English and Arabic. "
            "Start with the English answer, then add the Arabic translation below it under a '---' separator."
        )
    return "Answer in English."


def process_pdfs(files, chunk_sz, chunk_ov):
    """Load, chunk, and embed all uploaded PDFs into ChromaDB."""
    embeddings = GoogleGenerativeAIEmbeddings(
        model="gemini-embedding-001",
        google_api_key=api_key,
    )

    all_chunks = []
    names = []
    total_pages = 0

    for uploaded in files:
        if uploaded.name in st.session_state.processed_files:
            continue  # skip already-loaded files

        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(uploaded.read())
            tmp_path = tmp.name

        loader = PyPDFLoader(tmp_path)
        docs = loader.load()
        total_pages += len(docs)

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_sz,
            chunk_overlap=chunk_ov,
        )
        chunks = splitter.split_documents(docs)
        # Tag each chunk with the original filename
        for c in chunks:
            c.metadata["source"] = uploaded.name
        all_chunks.extend(chunks)
        names.append(uploaded.name)

    if not all_chunks:
        return False, "All files already processed."

    # Build or extend vector store
    if st.session_state.vectorstore is None:
        vs = Chroma.from_documents(
            documents=all_chunks,
            embedding=embeddings,
            collection_name="gis_docs",
        )
    else:
        st.session_state.vectorstore.add_documents(all_chunks)
        vs = st.session_state.vectorstore

    # Update state
    st.session_state.vectorstore = vs
    st.session_state.processed_files.extend(names)
    st.session_state.total_chunks += len(all_chunks)
    st.session_state.stats["pdfs_loaded"] += len(names)

    return True, f"Processed {len(all_chunks)} chunks from {len(names)} file(s)."


if process_btn:
    if not uploaded_files:
        st.warning("⚠️ Please upload at least one PDF first.")
    else:
        with st.spinner("📚 Processing PDFs — embedding and indexing..."):
            ok, msg = process_pdfs(uploaded_files, chunk_size, chunk_overlap)
        if ok:
            st.success(f"✅ {msg}")
        else:
            st.info(f"ℹ️ {msg}")
        st.rerun()


# ─────────────────────────────────────────────
# Loaded files display
# ─────────────────────────────────────────────
if st.session_state.processed_files:
    badges = " ".join(
        f'<span class="file-badge">📄 {f}</span>'
        for f in st.session_state.processed_files
    )
    st.markdown(f"<div style='margin-bottom:12px'>{badges}</div>", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# Chat Area
# ─────────────────────────────────────────────
if st.session_state.vectorstore is None:
    st.markdown("""
    <div style="background:#161b22;border:1px dashed #30363d;border-radius:12px;
                padding:40px;text-align:center;color:#484f58;margin-top:20px">
        <div style="font-size:40px;margin-bottom:12px">📂</div>
        <div style="font-size:16px;color:#8b949e">Upload a GIS PDF and click <b style="color:#58a6ff">Process PDFs</b> to start chatting.</div>
        <div style="font-size:12px;margin-top:8px;color:#484f58">
            Suggested PDFs: QGIS User Guide · ArcGIS Pro Docs · OGC Standards · EPSG Dataset
        </div>
    </div>
    """, unsafe_allow_html=True)
    st.stop()


# ─── Render past messages ───
chat_container = st.container()
with chat_container:
    for msg in st.session_state.messages:
        if msg["role"] == "user":
            st.markdown(f"""
            <div class="msg-user">
                <div class="msg-role">You · {msg.get('timestamp','')}</div>
                {msg['content']}
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="msg-assistant">
                <div class="msg-role">🗺️ GIS Assistant · {msg.get('timestamp','')}</div>
                {msg['content'].replace(chr(10), '<br>')}
            </div>
            """, unsafe_allow_html=True)

            # Sources
            sources = msg.get("sources", [])
            if sources:
                seen = set()
                pills = ""
                for doc in sources:
                    key = f"{doc.metadata.get('source','?')} p.{doc.metadata.get('page','?')}"
                    if key not in seen:
                        seen.add(key)
                        pills += f'<span class="source-pill">📄 {key}</span>'
                st.markdown(f"<div style='margin-top:6px;margin-bottom:4px'>{pills}</div>", unsafe_allow_html=True)

                with st.expander("🔍 View source excerpts"):
                    for i, doc in enumerate(sources, 1):
                        src = doc.metadata.get("source", "?")
                        pg = doc.metadata.get("page", "?")
                        st.markdown(f"""
                        <div style="background:#0d1117;border-left:3px solid #1f6feb;
                                    border-radius:4px;padding:10px 14px;margin:6px 0;
                                    font-size:12px;font-family:'IBM Plex Mono',monospace;color:#8b949e">
                            <b style="color:#58a6ff">[{i}] {src} — Page {pg}</b><br>
                            {doc.page_content[:300]}{'...' if len(doc.page_content) > 300 else ''}
                        </div>
                        """, unsafe_allow_html=True)


# ─── Input ───
st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)
question = st.chat_input("Ask anything about your GIS documents...")

if question:
    ts = datetime.datetime.now().strftime("%H:%M")

    # Save user message
    st.session_state.messages.append({
        "role": "user",
        "content": question,
        "sources": [],
        "timestamp": ts,
    })
    st.session_state.stats["questions_asked"] += 1

    with st.spinner("🤔 Searching documents and generating answer..."):
        # Retrieve
        docs = st.session_state.vectorstore.similarity_search(question, k=4)

        # Track pages sourced
        page_keys = set()
        for d in docs:
            key = f"{d.metadata.get('source','')}_{d.metadata.get('page','')}"
            if key not in page_keys:
                page_keys.add(key)
                st.session_state.stats["pages_read"] += 1

        context = "\n\n".join([d.page_content for d in docs])
        lang_instruction = build_language_instruction(st.session_state.language)

        prompt = f"""You are a GIS expert assistant. Use the document context below to answer the question.
If the answer isn't in the context, say so honestly — do NOT make up information.
Always reference which part of the documents you used.
{lang_instruction}

--- DOCUMENT CONTEXT ---
{context}
--- END CONTEXT ---

Question: {question}

Answer:"""

        llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            google_api_key=api_key,
            temperature=0.2,
        )
        response = llm.invoke(prompt)
        answer = response.content

    # Save assistant message
    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
        "sources": docs,
        "timestamp": datetime.datetime.now().strftime("%H:%M"),
    })

    st.rerun()
