# DocReader

A Retrieval-Augmented Generation (RAG) tool that lets you upload your own documents and ask questions about them — answered strictly from what's actually in the document, not from the model's general knowledge. Built entirely on Google's free Gemini API tier, with a Streamlit chat interface.

**Live demo:** (https://docreader-app-rishi.streamlit.app/)
> Note: hosted on Streamlit Community Cloud's free tier, which sleeps after 12 hours of inactivity. If the link shows a "waking up" screen, click through — it'll be live again within a minute.

## How it works

1. A document (`.md` file) is uploaded through the app's sidebar
2. The text is split into small, overlapping chunks
3. Each chunk is embedded using Gemini's embedding model and stored in a local Chroma vector database
4. When a question is asked, the app embeds the question, retrieves the most semantically similar chunks, and feeds them to Gemini as context
5. Gemini generates an answer grounded only in that retrieved context — and the app shows which source it came from

## Project structure

- `create_database.py` — ingestion pipeline: loads documents, chunks them, embeds and stores them in Chroma
- `query_data.py` — retrieval + generation pipeline: given a question, returns a grounded answer and its source
- `app.py` — Streamlit frontend: chat interface, file upload, and live knowledge-base rebuilding

## Setup — after cloning

**1. Create a virtual environment**
```bash
python3 -m venv venv
source venv/bin/activate      # Mac/Linux
venv\Scripts\activate         # Windows
```
Create the venv *after* the project folder has its final name and location — virtual environments hardcode absolute paths and break if the folder is later moved or renamed.

**2. Install dependencies**
```bash
pip install -r requirements.txt
```

**3. Download required NLTK data (one-time)**
```bash
python -c "import nltk; nltk.download('punkt_tab'); nltk.download('averaged_perceptron_tagger_eng')"
```

**4. Get a free Gemini API key**
Go to [Google AI Studio](https://aistudio.google.com/apikey), sign in, and create a key.

**5. Add it to a `.env` file** in the project root:
```
GOOGLE_API_KEY=your-key-here
```

**6. Run it**
```bash
streamlit run app.py
```
Or use the terminal-only workflow:
```bash
python create_database.py
python query_data.py "your question"
```

## Making it your own

This project isn't tied to any specific subject matter — any `.md` file can be uploaded and queried. A few things worth tuning if you repurpose it for very different content:

- **Chunk size** (`create_database.py`, `split_text()`): `300` works well for narrative/prose. Denser technical content often benefits from smaller chunks.
- **Number of retrieved chunks** (`query_data.py`, `k=3`): increase if answers feel incomplete or miss details spread across multiple chunks.
- **Relevance threshold** (`query_data.py`, `RELEVANCE_THRESHOLD`): this is tuned specifically for Gemini's embedding score range (roughly 0.4–0.6 for genuinely relevant matches) — it is *not* the same scale OpenAI's embeddings use, so don't reuse a threshold from another tutorial without re-checking actual scores first.
- **Prompt template** (`query_data.py`, `PROMPT_TEMPLATE`): rewrite for a different tone or response format.

## Deployment

Deployed via [Streamlit Community Cloud](https://share.streamlit.io) (free). Requires:
- `GOOGLE_API_KEY` set under the app's **Secrets** in Streamlit Cloud's dashboard (not committed to the repo)
- `requirements.txt` listing only direct dependencies, unpinned where possible, so pip can resolve versions compatible with whatever Python version the host runs

---

## Key hurdles hit while building this, and how they were resolved

This project started as a fork of a tutorial built around OpenAI's API and an older LangChain version. Getting it fully working — and then deployed — surfaced several non-obvious problems worth documenting.

### 1. Migrating from OpenAI to Gemini
The original tutorial used `OpenAIEmbeddings` and `ChatOpenAI`. Switching to Gemini required swapping in `GoogleGenerativeAIEmbeddings` and `ChatGoogleGenerativeAI` from `langchain-google-genai`, and critically, using the *same* embedding model consistently between database creation and querying — mixing embedding models silently breaks similarity search rather than throwing an error. Gemini's older free embedding model (`models/embedding-001`) was deprecated mid-project and had to be swapped to `gemini-embedding-001`.

### 2. LangChain version conflicts
The tutorial pinned an old `langchain==0.2.2`. Installing newer packages like `langchain-google-genai` pulled in a much newer `langchain-core`, which broke old import paths (`langchain.schema`, `langchain.prompts`) that had since been restructured. Fixed by importing directly from the current locations (`langchain_core.documents`, `langchain_core.prompts`) instead of the legacy wrapper modules, and by not hard-pinning `langchain`/`langchain-community` versions.

### 3. Chroma vector store deprecation and NumPy 2.0 incompatibility
`Chroma` imported from `langchain_community.vectorstores` is deprecated in favor of a dedicated `langchain-chroma` package. Sticking with the old import combined with a newer `chromadb`/NumPy pairing caused `AttributeError: np.float_ was removed in NumPy 2.0`. Fixed by installing `langchain-chroma` and importing `Chroma` from there instead.

### 4. Gemini free-tier rate limits during embedding
Embedding all chunks in one pass exceeded the free tier's 100 requests/minute limit and crashed mid-run. Fixed by embedding in small batches (25 chunks) with a pause between batches, staying safely under the limit.

### 5. NLTK data and SSL certificate errors
`unstructured` (used for markdown parsing) requires NLTK data (`punkt_tab`, `averaged_perceptron_tagger_eng`) downloaded on first use. On macOS, the python.org Python build doesn't ship trusted SSL certificates by default, causing `CERTIFICATE_VERIFY_FAILED` errors on that download. Fixed by running Python's certificate installer script once, then retrying the NLTK download.

### 6. Read-only filesystem on Streamlit Cloud
Locally, the app's own project folder is writable. On Streamlit Cloud, the deployed source folder is mounted read-only — writing the Chroma database to a relative path inside it failed with a database write error. Fixed by writing all generated data (uploaded files, the vector database) to the system's temp directory instead of the app's source folder.

### 7. "Readonly database" errors on repeated rebuilds
Even after moving to a writable temp directory, repeatedly deleting and recreating the *same* database folder on each rebuild intermittently produced `attempt to write a readonly database` errors — likely due to the old database connection not being fully released before deletion in a container environment. Fixed by generating a uniquely named database directory on every rebuild (via a UUID) and tracking the current active one through a small pointer file, rather than reusing one fixed path.

### 8. Dependency resolution across different Python versions
Using `pip freeze` to generate `requirements.txt` captured exact version pins from a local Python 3.12/3.13 environment. Streamlit Cloud's build environment ran Python 3.14, where some of those exact pinned versions (e.g. `spacy==3.8.14`, which was never actually published) didn't exist, breaking the entire install. Fixed by rewriting `requirements.txt` to list only direct imports, unpinned, letting pip resolve compatible versions itself for whatever Python version the host uses.

### 9. spaCy model not installable via normal pip resolution
`unstructured` relies on a spaCy language model (`en_core_web_sm`) that isn't distributed as a normal indexed PyPI package, so `pip install en_core_web_sm==3.8.0` fails outright, and `unstructured`'s own attempt to auto-download it at runtime failed silently in the cloud sandbox. Fixed by installing it as a direct wheel URL dependency in `requirements.txt`, so it's available at build time rather than fetched live.

## Known limitations

- **Shared, single-instance state**: all visitors to the deployed app currently share the same uploaded document and knowledge base — it isn't per-user/session-isolated. Fine for a demo; would need per-session storage for true multi-user use.
- **Non-persistent storage**: since the database lives in the host's temp directory, it does not survive an app restart/sleep cycle. A fresh upload is needed after the app wakes from sleep.
- **Markdown only**: currently only `.md` files are supported for upload.