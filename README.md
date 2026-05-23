# 🗺️ GIS Document Assistant

> A RAG-powered chatbot that lets you **chat with your GIS documents** in English or Arabic.  
> Built with LangChain · ChromaDB · Google Gemini · Streamlit

---

## 📸 Overview

This app lets you upload any GIS-related PDF (standards, manuals, tutorials) and ask questions about it in natural language. Instead of searching through hundreds of pages manually, the assistant retrieves the exact relevant sections and generates a grounded answer — with source citations.

Built as the **Day 4 Lab Project** for the ITI GIS Track · Generative AI Course.

---

## ✨ Features

| Feature | Description |
|---|---|
| 📄 Multi-PDF Upload | Load multiple GIS PDFs at once |
| 🔍 RAG-Powered Answers | Retrieves relevant chunks before generating — no hallucinations |
| 📚 Source Citations | Every answer shows which document and page it came from |
| 🌐 Arabic / Bilingual Mode | Answers in English, Arabic, or both |
| 💾 Export Chat to JSON | Download the full conversation with sources |
| 📊 Usage Statistics | Tracks questions asked, pages sourced, PDFs loaded |

---

## 🧠 How RAG Works

```
PDF ──► Chunk ──► Embed ──► ChromaDB
                                │
Question ──► Embed ──► Search ──┘
                          │
                    Relevant Chunks
                          │
          Question + Chunks ──► Gemini LLM ──► Answer
```

1. **Ingest** — PDFs are split into overlapping text chunks
2. **Embed** — Each chunk is converted to a vector using `gemini-embedding-001`
3. **Store** — Vectors are stored in ChromaDB (local vector database)
4. **Retrieve** — At query time, the question is embedded and the most similar chunks are fetched
5. **Generate** — Gemini receives the question + retrieved chunks and produces a grounded answer

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| LLM | Google Gemini 2.5 Flash |
| Embeddings | Google `gemini-embedding-001` (768 dimensions) |
| Vector DB | ChromaDB (local) |
| RAG Framework | LangChain |
| PDF Parsing | PyPDF + LangChain Community |
| UI | Streamlit |
| Language | Python 3.10+ |

---

## 🚀 Getting Started

### 1. Clone the repo

```bash
git clone https://github.com/MohamedElbalahy/gis-document-assistant.git
cd gis-document-assistant
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Get a Gemini API key

Go to [aistudio.google.com/apikey](https://aistudio.google.com/apikey) and create a free API key.

### 4. Run the app

```bash
streamlit run gis_rag_app.py
```

The app opens at `http://localhost:8501`. Enter your API key in the sidebar, upload a PDF, and start chatting.

---

## 📂 Project Structure

```
gis-document-assistant/
├── gis_rag_app.py       # Main Streamlit application
├── requirements.txt     # Python dependencies
└── README.md
```

---

## 💡 Suggested GIS PDFs to Try

- **QGIS User Guide** — [docs.qgis.org](https://docs.qgis.org)
- **ArcGIS Pro Documentation** — from Esri training materials
- **OGC Standards** — GeoJSON, KML, WMS specs
- **EPSG Geodetic Parameter Dataset** documentation

---

## 🗂️ Example Questions

```
What coordinate system does GPS use?
ما هو الفرق بين بيانات Raster و Vector؟
How do I perform a buffer analysis in QGIS?
What are the main file formats for vector data?
Explain the difference between WGS84 and Web Mercator.
```

---

## 🔮 Possible Extensions

- **Persistent vector store** — save ChromaDB to disk so PDFs don't need re-processing on restart
- **Confidence scores** — show similarity scores alongside each source citation
- **Metadata filters** — search within a specific PDF or page range only
- **Conversation memory** — pass chat history to the LLM for follow-up questions

---

## 👤 Author

**Mohamed Elbalahy**  
Civil & Environmental Engineer · GIS Specialist  
ITI GIS Track · Generative AI Course — 2025  

[![GitHub](https://img.shields.io/badge/GitHub-MohamedElbalahy-181717?logo=github)](https://github.com/MohamedElbalahy)

---

## 📄 License

MIT — free to use and modify.
