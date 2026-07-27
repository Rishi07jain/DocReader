import streamlit as st
from query_data import get_answer

st.set_page_config(
    page_title="DocReader",
    page_icon="📖",
    layout="centered",
)

# ---- Custom styling ----
st.markdown("""
    <style>
    .main {
        background-color: #0f1117;
    }
    .stChatMessage {
        border-radius: 12px;
    }
    h1 {
        font-weight: 700;
        letter-spacing: -0.5px;
    }
    .source-tag {
        font-size: 0.75rem;
        color: #8b8f98;
        margin-top: 4px;
    }
    </style>
""", unsafe_allow_html=True)

st.title("📖 DocReader")
st.caption("Ask questions about your documents — answered using only what's actually in them.")

# ---- Session state for chat history ----
if "messages" not in st.session_state:
    st.session_state.messages = []

# ---- Render existing chat history ----
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