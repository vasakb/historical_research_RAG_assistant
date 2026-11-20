# 📜 Historical RAG Assistant

A beginner-friendly Retrieval-Augmented Generation (RAG) assistant designed for exploring and querying historical documents, manuscripts, and archives.

## 🗂️ Project Structure

```text
historical-rag-assistant/
├── data/                  # Directory to store uploaded or sample historical PDFs
├── vectorstore/           # Directory where persisted FAISS vector indexes are stored
├── src/                   # Core Python modules for RAG pipeline components
│   ├── __init__.py        # Makes src a Python package
│   ├── pdf_loader.py      # 1. PDF processing using PyPDF
│   ├── text_chunker.py    # 2. Document chunking using LangChain text splitters
│   ├── embeddings.py      # 3. Embedding model initialization via HuggingFace
│   ├── vector_db.py       # 4. FAISS vector store creation, saving, and loading
│   └── qa_chain.py        # 5. Question answering pipeline with Groq LLM
├── app.py                 # 6. Streamlit interactive web frontend
├── requirements.txt       # Project dependencies
├── .env.example           # Example template for environment secrets (Groq API Key)
└── .gitignore             # Files and directories ignored by Git
```

## 🚀 Getting Started

### 1. Create and Activate a Virtual Environment

On macOS / Linux:
```bash
python3 -m venv venv
source venv/bin/activate
```

On Windows:
```cmd
python -m venv venv
venv\Scripts\activate
```

### 2. Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 3. Configure Your Environment Variables

Create your local `.env` file from `.env.example`:
```bash
cp .env.example .env
```
Open `.env` and paste your Groq API key:
```env
GROQ_API_KEY=your_groq_api_key_here
```
*(You can get a free Groq API key at [console.groq.com](https://console.groq.com/keys)).*

### 4. Run the Streamlit Application

```bash
streamlit run app.py
```

