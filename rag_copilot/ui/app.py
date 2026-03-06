import os
import requests
import streamlit as st

API_URL = os.getenv("API_URL", "http://localhost:8000")

st.set_page_config(page_title="Regulatory Copilot", layout="wide")


# Session state

if "doc_id" not in st.session_state:
    st.session_state.doc_id = None
if "filename" not in st.session_state:
    st.session_state.filename = None
if "is_indexed" not in st.session_state:
    st.session_state.is_indexed = False
if "messages" not in st.session_state:
    st.session_state.messages = []
if "last_citations" not in st.session_state:
    st.session_state.last_citations = []


# API helpers

def ingest_pdf(file):
    files = {"file": (file.name, file.getvalue(), "application/pdf")}
    r = requests.post(f"{API_URL}/ingest/pdf", files=files, timeout=300)
    r.raise_for_status()
    return r.json()

def index_all(doc_id):
    rq = requests.post(
        f"{API_URL}/index/qdrant",
        json={"doc_id": doc_id, "batch_size": 64},
        timeout=300,
    )
    rq.raise_for_status()

    rb = requests.post(
        f"{API_URL}/index/bm25",
        json={"doc_id": doc_id, "batch_size": 200},
        timeout=300,
    )
    rb.raise_for_status()

    return {"qdrant": rq.json(), "bm25": rb.json()}

def chat(question, doc_id, alpha, top_k):
    payload = {
        "question": question,
        "doc_id": doc_id,
        "alpha": alpha,
        "top_k": top_k,
    }
    r = requests.post(f"{API_URL}/chat", json=payload, timeout=300)
    r.raise_for_status()
    return r.json()


# Ready check

def is_ready():
    return bool(st.session_state.doc_id) and st.session_state.is_indexed


# Header

st.title("📄 Regulatory Copilot")
st.caption("Upload a PDF and ask questions. Answers include citations so you can verify them.")

if is_ready():
    st.success(
        f"Ready ✅  Document: {st.session_state.filename}  |  doc_id: {st.session_state.doc_id}"
    )
else:
    st.warning(
        "Not ready yet ⚠️  Step 1: Upload + Ingest. Step 2: Index. Step 3: Ask questions."
    )


# Sidebar

with st.sidebar:

    st.header("Controls")

    alpha = st.slider(
        "Answer style (semantic vs keyword)",
        0.0,
        1.0,
        0.6,
        0.05,
        help="Higher = more semantic meaning. Lower = more keyword matching.",
    )

    top_k = st.slider("Evidence chunks used", 2, 12, 6, 1)

    st.divider()

    st.subheader("Step 1 — Upload a PDF")

    uploaded = st.file_uploader("Choose a PDF", type=["pdf"])

    st.subheader("Step 2 — Process the document")

    auto_index = st.checkbox(
        "Auto-index after ingest (recommended)",
        value=True,
    )

    col1, col2 = st.columns(2)

    ingest_btn = col1.button(
        "Ingest",
        use_container_width=True,
        disabled=uploaded is None,
    )

    index_btn = col2.button(
        "Index",
        use_container_width=True,
        disabled=st.session_state.doc_id is None,
    )

    st.divider()

    st.subheader("Session")

    if st.button("Clear chat", use_container_width=True):
        st.session_state.messages = []
        st.session_state.last_citations = []
        st.toast("Chat cleared")

    if st.button("Reset document", use_container_width=True):
        st.session_state.doc_id = None
        st.session_state.filename = None
        st.session_state.is_indexed = False
        st.session_state.messages = []
        st.session_state.last_citations = []
        st.toast("Document reset")
        st.rerun()


# Ingest logic

if ingest_btn and uploaded is not None:

    with st.spinner("Ingesting PDF..."):

        meta = ingest_pdf(uploaded)

        st.session_state.doc_id = meta["doc_id"]
        st.session_state.filename = meta["filename"]
        st.session_state.is_indexed = False

    st.success(f"Ingested ✅ doc_id = {st.session_state.doc_id}")

    if auto_index:

        with st.spinner("Indexing (Qdrant + BM25)..."):

            _ = index_all(st.session_state.doc_id)

            st.session_state.is_indexed = True

        st.success("Indexed ✅ You can now chat.")

    st.rerun()


# Manual index logic

if index_btn and st.session_state.doc_id is not None:

    with st.spinner("Indexing (Qdrant + BM25)..."):

        _ = index_all(st.session_state.doc_id)

        st.session_state.is_indexed = True

    st.success("Indexed ✅ You can now chat.")

    st.rerun()


# Layout

left, right = st.columns([2, 1], gap="large")


# Chat column

with left:

    st.subheader("Chat")

    for m in st.session_state.messages:

        with st.chat_message(m["role"]):
            st.write(m["content"])

    user_text = st.chat_input(
        "Ask a question about the document…",
        disabled=not is_ready(),
    )

    if user_text and is_ready():

        st.session_state.messages.append(
            {"role": "user", "content": user_text}
        )

        with st.chat_message("user"):
            st.write(user_text)

        with st.chat_message("assistant"):

            with st.spinner("Generating answer..."):

                data = chat(
                    user_text,
                    st.session_state.doc_id,
                    alpha,
                    top_k,
                )

            answer = data["answer"]

            st.write(answer)

        st.session_state.messages.append(
            {"role": "assistant", "content": answer}
        )

        st.session_state.last_citations = data.get("citations", [])


# Citation column

with right:

    st.subheader("Citations")

    cites = st.session_state.last_citations

    if not cites:

        st.caption("Citations will appear here after you ask a question.")

    else:

        st.caption("Click a citation to view the supporting excerpt.")

        for c in cites:

            title = f"{c.get('filename')} — page {c.get('page')} — {c.get('chunk_id')}"

            with st.expander(title, expanded=False):

                st.write(c.get("snippet", ""))