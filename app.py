"""Phase 7: Streamlit interface. Run with `streamlit run app.py`."""
import streamlit as st

from rag import DATA_DIR, answer_question, build_vectorstore, format_source, get_llm, load_vectorstore

st.set_page_config(page_title="PDF Chatbot (RAG)", page_icon="📄")
st.title("📄 Chat with your PDFs")

DATA_DIR.mkdir(exist_ok=True)

# ---- Sidebar: upload + process ----
with st.sidebar:
    st.header("Documents")
    uploads = st.file_uploader("Upload PDFs", type="pdf", accept_multiple_files=True)
    if st.button("Process documents", type="primary"):
        if uploads:
            for f in uploads:
                (DATA_DIR / f.name).write_bytes(f.getbuffer())
        try:
            with st.spinner("Reading, chunking and embedding..."):
                n = build_vectorstore()
            st.session_state.pop("store", None)
            st.session_state.messages = []
            st.success(f"Ready! Indexed {n} chunks.")
        except ValueError as e:
            st.error(str(e))
    if st.button("Clear chat"):
        st.session_state.messages = []

# ---- Load resources once ----
if "messages" not in st.session_state:
    st.session_state.messages = []
if "store" not in st.session_state:
    try:
        st.session_state.store = load_vectorstore()
    except FileNotFoundError:
        st.session_state.store = None
if "llm" not in st.session_state:
    try:
        st.session_state.llm = get_llm()
    except Exception as e:
        st.session_state.llm = None
        st.error(f"LLM setup problem: {e}")

if st.session_state.store is None:
    st.info("Upload PDFs in the sidebar and click **Process documents** to begin.")
    st.stop()

# ---- Chat history ----
for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])
        if m.get("sources"):
            with st.expander("Sources"):
                for s in m["sources"]:
                    st.write("•", s)

# ---- New question ----
if question := st.chat_input("Ask something about your documents"):
    history = [{"role": m["role"], "content": m["content"]} for m in st.session_state.messages]
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        if st.session_state.llm is None:
            st.error("No LLM configured. Check your .env file.")
            st.stop()
        with st.spinner("Thinking..."):
            answer, chunks = answer_question(
                st.session_state.store, st.session_state.llm, question, history
            )
        sources = sorted({format_source(c) for c in chunks})
        st.markdown(answer)
        with st.expander("Sources"):
            for s in sources:
                st.write("•", s)
    st.session_state.messages.append({"role": "assistant", "content": answer, "sources": sources})
