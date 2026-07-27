import argparse
from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv

load_dotenv()

CHROMA_PATH = "chroma"
RELEVANCE_THRESHOLD = 0.5  # tuned for Gemini's embedding score range

PROMPT_TEMPLATE = """
Answer the question based only on the following context:

{context}

---

Answer the question based on the above context: {question}
"""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("query_text", type=str, help="The query text.")
    args = parser.parse_args()
    query_text = args.query_text

    embedding_function = GoogleGenerativeAIEmbeddings(model="gemini-embedding-001")
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)

    results = db.similarity_search_with_relevance_scores(query_text, k=3)

    if len(results) == 0 or results[0][1] < RELEVANCE_THRESHOLD:
        print("I couldn't find anything relevant to answer that question.")
        return

    context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context_text, question=query_text)

    model = ChatGoogleGenerativeAI(model="gemini-2.5-flash")
    response_text = model.invoke(prompt).content

    source_files = sorted(set(doc.metadata.get("source", "Unknown") for doc, _score in results))

    print(f"\n{response_text}\n")
    print(f"Source: {', '.join(source_files)}")


if __name__ == "__main__":
    main()