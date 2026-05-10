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
def retrieval_debug(query, doc_id, alpha, top_k_vec=8, top_k_bm25=8, top_k_hybrid=8, use_rewrite=True):
    payload = {
        "query": query,
        "doc_id": doc_id,
        "top_k_vec": top_k_vec,
        "top_k_bm25": top_k_bm25,
        "top_k_hybrid": top_k_hybrid,
        "alpha": alpha,
        "use_rewrite": use_rewrite,
    }

    r = requests.post(f"{API_URL}/retrieve/debug", json=payload, timeout=300)
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
st.divider()

with st.expander("🔎 Debug retrieval", expanded=False):

    st.caption(
        "Inspect what the retriever is doing before the final answer is generated."
    )

    debug_query = st.text_input(
        "Debug query",
        placeholder="Enter a question to inspect retrieval...",
        disabled=not is_ready(),
    )

    col_a, col_b, col_c = st.columns(3)

    top_k_vec = col_a.slider("Vector top-k", 1, 20, 8, 1)
    top_k_bm25 = col_b.slider("BM25 top-k", 1, 20, 8, 1)
    top_k_hybrid = col_c.slider("Hybrid top-k", 1, 20, 8, 1)

    use_rewrite = st.checkbox("Use query rewriting", value=True)

    run_debug = st.button(
        "Run retrieval debug",
        disabled=not is_ready() or not debug_query.strip(),
        use_container_width=True,
    )

    if run_debug:

        with st.spinner("Running retrieval debug..."):

            debug_data = retrieval_debug(
                query=debug_query,
                doc_id=st.session_state.doc_id,
                alpha=alpha,
                top_k_vec=top_k_vec,
                top_k_bm25=top_k_bm25,
                top_k_hybrid=top_k_hybrid,
                use_rewrite=use_rewrite,
            )

        st.success("Retrieval debug completed ✅")

        st.subheader("Rewritten query")

        rewritten = debug_data.get("rewritten", {})

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**Semantic query**")
            st.code(rewritten.get("sem", ""), language="text")

        with col2:
            st.markdown("**Keyword query**")
            st.code(rewritten.get("kw", ""), language="text")

        def render_chunks(title, chunks):
            st.subheader(title)

            if not chunks:
                st.info("No chunks returned.")
                return

            for i, chunk in enumerate(chunks, start=1):
                score = chunk.get("score", chunk.get("hybrid_score", "N/A"))
                page = chunk.get("page", "N/A")
                chunk_id = chunk.get("chunk_id", "N/A")
                filename = chunk.get("filename", "Unknown file")

                expander_title = (
                    f"#{i} | score: {score} | page: {page} | chunk: {chunk_id}"
                )

                with st.expander(expander_title, expanded=i <= 2):
                    st.markdown(f"**File:** {filename}")
                    st.markdown(f"**Page:** {page}")
                    st.markdown(f"**Chunk ID:** `{chunk_id}`")

                    if "vector_score" in chunk or "bm25_score" in chunk:
                        st.markdown(
                            f"""
                            **Vector score:** `{chunk.get("vector_score", "N/A")}`  
                            **BM25 score:** `{chunk.get("bm25_score", "N/A")}`  
                            **Hybrid score:** `{chunk.get("hybrid_score", "N/A")}`
                            """
                        )

                    text = (
                        chunk.get("text")
                        or chunk.get("snippet")
                        or chunk.get("content")
                        or ""
                    )

                    st.write(text)

        tab1, tab2, tab3 = st.tabs(
            ["Vector results", "BM25 results", "Hybrid results"]
        )

        with tab1:
            render_chunks(
                "Top vector chunks",
                debug_data.get("vector_results", []),
            )

        with tab2:
            render_chunks(
                "Top BM25 chunks",
                debug_data.get("bm25_results", []),
            )

        with tab3:
            render_chunks(
                "Top hybrid chunks",
                debug_data.get("hybrid_results", []),
            )


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