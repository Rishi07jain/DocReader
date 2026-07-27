from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
import os
import streamlit as st

# Load environment variables from a local .env file (used when running locally).
load_dotenv()

# Fallback: if GOOGLE_API_KEY isn't set via .env (e.g. when running on
# Streamlit Cloud, where .env doesn't exist), pull it from Streamlit's
# built-in secrets manager instead.
if "GOOGLE_API_KEY" not in os.environ:
    try:
        os.environ["GOOGLE_API_KEY"] = st.secrets["GOOGLE_API_KEY"]
    except (KeyError, FileNotFoundError):
        pass

# Path to the local Chroma vector database created by create_database.py
CHROMA_PATH = "chroma"

# Minimum similarity score a retrieved chunk must have to be considered
# "relevant enough" to answer from. Tuned for Gemini's embedding score
# range (0.4-0.5), which is NOT the same scale OpenAI's embeddings use.
RELEVANCE_THRESHOLD = 0.5

# Template used to build the final prompt sent to the chat model.
# The model is instructed to answer strictly from the retrieved context,
# which is what keeps answers grounded instead of hallucinated.
PROMPT_TEMPLATE = """
Answer the question based only on the following context:

{context}

---

Answer the question based on the above context: {question}
"""


@st.cache_resource
def get_embedding_function():
    """
    Creates (once) and reuses the Gemini embedding client.
    Cached so it isn't recreated on every single question/rerun,
    which was adding unnecessary latency to each query.
    """
    return GoogleGenerativeAIEmbeddings(model="gemini-embedding-001")


@st.cache_resource
def get_chat_model():
    """
    Creates (once) and reuses the Gemini chat client.
    Same caching reasoning as get_embedding_function().
    """
    return ChatGoogleGenerativeAI(model="gemini-2.5-flash")


def get_answer(query_text: str):
    """
    Runs the full RAG pipeline for a given question:
    1. Embed the question
    2. Retrieve the most relevant chunks from the vector database
    3. Feed those chunks + the question to the chat model as context
    4. Return the generated answer and which source file(s) it came from

    Returns:
        (response_text, source_files) on success
        (None, None) if no sufficiently relevant content was found
    """
    embedding_function = get_embedding_function()

    # Connect to the existing Chroma database on disk (built separately
    # by create_database.py or via the Streamlit upload feature).
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)

    # Retrieve the top 3 most semantically similar chunks to the question,
    # along with a relevance score for each.
    results = db.similarity_search_with_relevance_scores(query_text, k=3)

    # If nothing was found, or the best match isn't relevant enough,
    # don't attempt to answer — avoids confidently answering off-topic
    # questions with unrelated content.
    if len(results) == 0 or results[0][1] < RELEVANCE_THRESHOLD:
        return None, None

    # Combine the retrieved chunks into one context block, separated
    # clearly so the model can distinguish between source passages.
    context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])

    # Build the final prompt using the template, injecting the retrieved
    # context and the original question.
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context_text, question=query_text)

    # Generate the answer using the cached chat model.
    model = get_chat_model()
    response_text = model.invoke(prompt).content

    # Collect and deduplicate the source file(s) the retrieved chunks came
    # from, so the same file isn't listed multiple times if several
    # chunks matched from it.
    source_files = sorted(set(doc.metadata.get("source", "Unknown") for doc, _score in results))

    return response_text, source_files


def main():
    """
    Command-line entry point, allowing this file to still be run directly
    from the terminal (e.g. `python query_data.py "your question"`),
    independent of the Streamlit app.
    """
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("query_text", type=str, help="The query text.")
    args = parser.parse_args()

    response_text, source_files = get_answer(args.query_text)

    if response_text is None:
        print("I couldn't find anything relevant to answer that question.")
        return

    print(f"\n{response_text}\n")
    print(f"Source: {', '.join(source_files)}")


if __name__ == "__main__":
    main()