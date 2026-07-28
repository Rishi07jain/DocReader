import streamlit as st
import os
from query_data import get_answer, get_chroma_path
from create_database import generate_data_store, DATA_PATH


st.set_page_config(
    page_title="DocReader",
    page_icon="📖",
    layout="centered",
)

st.markdown("""
    <style>
    .main { background-color: #0f1117; }
    .stChatMessage { border-radius: 12px; }
    h1 { font-weight: 700; letter-spacing: -0.5px; }
    .source-tag { font-size: 0.75rem; color: #8b8f98; margin-top: 4px; }
    </style>
""", unsafe_allow_html=True)

st.title("📖 DocReader")
st.caption("Upload a document, then ask questions about it — answered using only what's actually in it.")

# ---- Sidebar: document upload ----
with st.sidebar:
    st.header("📄 Documents")
    uploaded_file = st.file_uploader("Upload a markdown (.md) file", type=["md"])
    replace_existing = st.checkbox("Replace existing documents", value=True)

    if uploaded_file is not None:
        if st.button("Build knowledge base from this file"):
            os.makedirs(DATA_PATH, exist_ok=True)

            if replace_existing:
                for f in os.listdir(DATA_PATH):
                    if f.endswith(".md"):
                        os.remove(os.path.join(DATA_PATH, f))

            file_path = os.path.join(DATA_PATH, uploaded_file.name)
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

            with st.spinner("Reading, chunking, and embedding your document... this can take a few minutes for larger files."):
                generate_data_store()

            st.success(f"Knowledge base built from {uploaded_file.name}")
            st.session_state.messages = []  # clear old chat since context changed

    st.divider()
    if os.path.exists(DATA_PATH):
        current_files = [f for f in os.listdir(DATA_PATH) if f.endswith(".md")]
        if current_files:
            st.caption("Currently loaded:")
            for f in current_files:
                st.caption(f"• {f}")

# ---- Session state for chat history ----
if "messages" not in st.session_state:
    st.session_state.messages = []

# ---- Render chat history ----
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message.get("sources"):
            st.markdown(
                f'<div class="source-tag">Source: {", ".join(message["sources"])}</div>',
                unsafe_allow_html=True,
            )

# ---- Chat input ----
query_text = st.chat_input("Ask something about your documents...")

if query_text:
    if get_chroma_path() is None:
        st.warning("Upload a document and build the knowledge base first (see sidebar).")
    else:
        st.session_state.messages.append({"role": "user", "content": query_text})
        with st.chat_message("user"):
            st.markdown(query_text)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response_text, sources = get_answer(query_text)

            if response_text is None:
                answer = "I couldn't find anything relevant to answer that question."
                st.markdown(answer)
                st.session_state.messages.append({"role": "assistant", "content": answer})
            else:
                st.markdown(response_text)
                st.markdown(
                    f'<div class="source-tag">Source: {", ".join(sources)}</div>',
                    unsafe_allow_html=True,
                )
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": response_text,
                    "sources": sources,
                })