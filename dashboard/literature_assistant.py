import streamlit as st
from rag.document_loader import SourceMeta, load_pdf
from rag.text_chunker import chunk_document
from rag.vector_store import VectorStore
from rag.qa_engine import answer
from dashboard.common import badge

TOPICS = ["sulfide electrolyte dendrite suppression critical current density", "solid electrolyte interfacial impedance and stack pressure",
          "sodium-ion layered oxide vs polyanionic vs Prussian blue cathodes", "sodium-ion low-temperature performance and cycle life"]


def render():
    st.header("RAG Literature Assistant")
    store = VectorStore()
    st.caption("Local TF-IDF retrieval over documents you upload. No paper is bundled, and no API key is required.")
    with st.expander("Add a permitted PDF (open-access or one you are allowed to use)"):
        f = st.file_uploader("PDF", type="pdf")
        title = st.text_input("Title (required)")
        c1, c2 = st.columns(2)
        authors, year = c1.text_input("Authors"), c2.text_input("Year")
        ident, url = c1.text_input("DOI / arXiv id"), c2.text_input("Source URL")
        if st.button("Index PDF") and f and title.strip():
            doc = load_pdf(f, SourceMeta(title.strip(), authors, year, ident, url))
            chunks = chunk_document(doc)
            if chunks:
                store.add(chunks)
                st.success(f"Indexed {len(chunks)} chunks from {len(doc.pages)} pages.")
            else:
                badge("bad", "No extractable text (scanned PDF?). Nothing indexed.")
        elif st.button("Index PDF (check inputs)", key="chk"):
            badge("warn", "Choose a PDF and enter a title.")
    srcs = store.sources()
    st.subheader(f"Indexed sources ({len(srcs)})")
    if srcs:
        st.dataframe(srcs, width="stretch", hide_index=True)
    q = st.text_input("Question", placeholder=TOPICS[0])
    if q:
        res = answer(store, q)
        badge("info" if res["status"] == "retrieved" else "warn", res["message"])
        for p in res["passages"]:
            s = p["source"]
            st.markdown(f"**[{p['rank']}] {s['title']}** - {s['authors'] or 'authors not provided'}, {s['year'] or 'year n/a'} "
                        f"| {s['identifier'] or 'no identifier'} | {s['url'] or 'no URL'} | page {p['page']} | score {p['score']}")
            st.text(p["excerpt"])
        if res.get("interpretation"):
            st.subheader("Model-generated interpretation (separate from retrieved text)")
            st.write(res["interpretation"])
