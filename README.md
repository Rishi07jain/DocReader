# RAG Tutorial with LangChain + Gemini

A simple Retrieval-Augmented Generation (RAG) pipeline that lets you ask questions about your own documents and get answers grounded in that content — no OpenAI key required, runs on Google's free Gemini API tier.

This started as a fork of [pixegami's LangChain RAG tutorial](https://github.com/pixegami/langchain-rag-tutorial), originally built around OpenAI's models. I swapped it over to Google Gemini (for both embeddings and the chat model), fixed a handful of version-compatibility issues along the way, and made it document-agnostic — drop in whatever `.md` files you want, it doesn't care what's in them.

## How it works

1. You put your source documents (markdown files) in `data/books/`
2. `create_database.py` loads them, splits them into small overlapping chunks, embeds each chunk using Gemini, and stores everything in a local Chroma vector database
3. `query_data.py` takes a question, finds the most relevant chunks from that vector database, stuffs them into a prompt as context, and asks Gemini to answer based only on that context

That's the whole idea behind RAG — instead of the model answering from what it was trained on, it answers from what you actually gave it.

## Setup

**1. Clone and enter the project**
```bash
git clone <your-repo-url>
cd langchain-rag-tutorial
```

**2. Create a virtual environment**
```bash
python -m venv venv
source venv/bin/activate      # Mac/Linux
venv\Scripts\activate         # Windows
```
> Avoid putting this project inside a folder path with special characters like `:` or `/` in the name — Python's venv tool will refuse to create an environment there.

**3. Install dependencies**
```bash
pip install -r requirements.txt
pip install "unstructured[md]"
pip install langchain-google-genai
```

**4. Get a free Gemini API key**
- Go to [Google AI Studio](https://aistudio.google.com/apikey)
- Sign in, click "Create API Key," copy it

**5. Add it to a `.env` file** in the project root:
```
GOOGLE_API_KEY=your-key-here
```
This file is already git-ignored, so it won't get pushed anywhere by accident.

**6. Download required NLTK data** (used internally by the document parser)
```bash
python -c "import nltk; nltk.download('punkt_tab'); nltk.download('averaged_perceptron_tagger_eng')"
```

## Usage

**Add your documents**

Drop any `.md` files into `data/books/`. Doesn't matter what they're about — a novel, your notes, product docs, whatever you want to be able to query later.

**Build the vector database**
```bash
python create_database.py
```
This reads everything in `data/books/`, chunks it, and embeds it via Gemini. For larger document sets, this is rate-limited to stay within Gemini's free tier (100 embedding requests/minute), so it may take a few minutes — that's expected, not a bug.

**Ask a question**
```bash
python query_data.py "your question here"
```
You'll get a response generated from the actual content of your documents, plus the source file(s) it pulled from.

## A few things I ran into (so you don't have to)

- **Old `langchain.schema` / `langchain.prompts` imports break** on newer `langchain-core` versions — use `langchain_core.documents` and `langchain_core.prompts` instead.
- **`models/embedding-001` is deprecated.** Use `gemini-embedding-001` for embeddings.
- **Free tier rate limits are real.** If you're embedding a lot of chunks at once, batch the requests with a short delay between batches instead of firing everything at once.
- **iCloud Drive can silently break local databases.** If your project folder lives inside `~/Documents` with iCloud sync on, you may hit `readonly database` errors mid-run. Moving the project outside of iCloud-synced folders fixes it.

## Notes

- The `chroma/` folder (your vector database) is regenerated every time you run `create_database.py`, so it's git-ignored — no need to commit it.
- Whatever embedding model you use to build the database, use the *same* one when querying, or the similarity search won't make sense.
