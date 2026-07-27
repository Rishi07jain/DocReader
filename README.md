# DocReader

A Retrieval-Augmented Generation (RAG) tool that lets you ask questions about your own documents and get answers grounded in that content — with a clean chat interface built on Streamlit. Runs entirely on Google's free Gemini API tier, no OpenAI key needed.

Drop in any markdown files, ask questions, and get answers pulled directly from what you gave it — not from the model's general training knowledge.

## How it works

1. `create_database.py` loads your documents from `data/books/`, splits them into small overlapping chunks, embeds each chunk using Gemini, and stores everything in a local Chroma vector database
2. `query_data.py` takes a question, finds the most relevant chunks from that database, feeds them to Gemini as context, and returns an answer grounded in that context — plus which source file it came from
3. `app.py` wraps that same logic in a Streamlit chat interface, so you can ask questions through a browser instead of the terminal

## Setup — after cloning

**1. Enter the project and create a virtual environment**
```bash
cd DocReader
python3 -m venv venv
source venv/bin/activate      # Mac/Linux
venv\Scripts\activate         # Windows
```
> Create the venv *after* your folder has its final name/location — venvs hardcode absolute paths and break if the folder is moved or renamed afterward.

**2. Install dependencies**
```bash
pip install -r requirements.txt
pip install "unstructured[md]"
```

**3. Download required NLTK data (one-time)**
```bash
python -c "import nltk; nltk.download('punkt_tab'); nltk.download('averaged_perceptron_tagger_eng')"
```

**4. Get a free Gemini API key**
- Go to [Google AI Studio](https://aistudio.google.com/apikey)
- Sign in, click "Create API Key," copy it

**5. Add it to a `.env` file** in the project root:
```
GOOGLE_API_KEY=your-key-here
```
This file is git-ignored, so it stays local and private.

**6. Build the vector database**
```bash
python create_database.py
```
This embeds everything currently in `data/books/`. For larger document sets this may take a few minutes — it's intentionally rate-limited to stay within Gemini's free tier (100 embedding requests/minute), so pausing partway through is expected, not a bug.

**7. Ask questions**

From the terminal:
```bash
python query_data.py "your question"
```

Or launch the chat interface:
```bash
streamlit run app.py
```
This opens the app in your browser at `http://localhost:8501`.

## Making it your own — swapping in new content

This project isn't tied to any particular subject matter. To point it at your own documents:

**1. Replace the source files**

Delete or add `.md` files inside `data/books/`. It can be anything — notes, articles, documentation, a book, transcripts — as long as it's markdown text.

> Currently only `.md` files are loaded (`DirectoryLoader(DATA_PATH, glob="*.md")` in `create_database.py`). If your source material is in another format:
> - **PDF**: change the loader to `PyPDFDirectoryLoader` (from `langchain_community.document_loaders`) and update the glob to `*.pdf`
> - **Plain text**: change the glob to `*.txt`
> - **Mixed formats**: `unstructured`'s `DirectoryLoader` can auto-detect multiple types if you don't restrict the glob pattern, but you'll want to test this against your actual files first

**2. Rebuild the database**
```bash
python create_database.py
```
This wipes the existing `chroma/` folder and re-embeds from scratch based on whatever is currently in `data/books/`. There's no incremental "add just this file" mode — every run is a full rebuild.

**3. Adjust retrieval settings if needed** (optional, in both `create_database.py` and `query_data.py`)

- **Chunk size** (`create_database.py`, in `split_text()`): `chunk_size=300` works well for narrative prose. Denser technical content (code, specs, structured data) often benefits from smaller chunks (150–250) so retrieval pulls back more precise, less noisy context.
- **Number of retrieved chunks** (`query_data.py`, `k=3` in `similarity_search_with_relevance_scores`): increase this (e.g. `k=5`) if answers feel too thin or miss relevant details spread across multiple chunks.
- **Relevance threshold** (`query_data.py`, `RELEVANCE_THRESHOLD = 0.5`): if you're getting "I couldn't find anything relevant" too often on questions that should be answerable, lower this slightly. If you're getting confident-sounding answers on totally unrelated questions, raise it. This value is tuned for Gemini's `gemini-embedding-001` score range — it is not the same scale OpenAI's embeddings use, so don't reuse a threshold from another OpenAI-based tutorial without re-checking it.
- **Prompt template** (`query_data.py`, `PROMPT_TEMPLATE`): rewrite this if you want a different tone, stricter grounding, or a different response format (e.g. bullet points instead of prose).

**4. Update the app's branding (optional)**

In `app.py`, the title, caption, and page icon in `st.set_page_config()` and `st.title()` are just cosmetic — change them to match whatever your new content actually is (e.g. "Course Notes Q&A" instead of "DocReader") if you want the interface to reflect the new subject matter.

## Notes

- `venv/`, `chroma/`, and `.env` are all git-ignored — they're meant to be recreated locally, not committed
- Whatever embedding model builds the database must match the one used to query it, or similarity search will return meaningless results
- The chat history in the Streamlit app only lives in memory for that browser session — closing or restarting the app clears it
