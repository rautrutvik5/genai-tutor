# ===== MEMBER C owns this file (UI) =====
import os, streamlit as st
import rag

st.set_page_config(page_title="OE GenAI Tutor", page_icon="🎓")
st.title("🎓 OE GenAI Tutor")

with st.sidebar:
    use_rag = st.toggle("Use retrieval (RAG)", value=True)
    k = st.slider("Top-k chunks", 1, 8, rag.TOP_K)
    if st.button("Clear chat"):
        st.session_state.history = []
    st.caption(f"Model: {rag.LLM} | Embeddings: {rag.EMB}")

if not os.path.exists("index/vecs.npy"):
    st.warning("Index not built. Run: python rag.py data/handbook.pdf")
    st.stop()

st.session_state.setdefault("history", [])
for h in st.session_state.history:
    with st.chat_message(h["role"]):
        st.write(h["content"])
        if h.get("chunks"):
            with st.expander("Sources used"):
                for c in h["chunks"]:
                    st.markdown(f"**p.{c['page']}** (score {c['score']:.2f})")
                    st.caption(c["text"])

if q := st.chat_input("Ask about the Generative AI course..."):
    st.session_state.history.append({"role": "user", "content": q})
    with st.spinner("Thinking..."):
        out = rag.answer(q, use_rag=use_rag, k=k)
    st.session_state.history.append({"role": "assistant", "content": out["answer"], "chunks": out["chunks"]})
    st.rerun()

if st.session_state.history:
    log = "\n\n".join(f"{h['role'].upper()}: {h['content']}" for h in st.session_state.history)
    st.sidebar.download_button("Export chat", log, "chat.txt")
