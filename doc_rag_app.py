import io
import os
import shutil
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_classic.evaluation.qa import QAEvalChain
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader


load_dotenv()

st.set_page_config(page_title="Document RAG + QAEval", page_icon="📚", layout="wide")
st.title("Document RAG + QAEval Dashboard")
st.caption("Upload documents, chunk with LangChain, store vectors in DB, query with RAG, and evaluate with QAEval")

from pathlib import Path
import os

# Writable location for Streamlit Community Cloud
VECTOR_DB_DIR = Path(os.getenv("CHROMA_DIR", "/tmp/vector_db_docs"))
COLLECTION_NAME = "doc_rag_collection"
DEFAULT_MODEL = "gpt-4.1-mini"
DEFAULT_EMBED_MODEL = "text-embedding-3-small"
MAX_UPLOAD_MB = int(os.getenv("MAX_UPLOAD_MB", "25"))



def resolve_api_key() -> str:
    secret_key = ""
    try:
        secret_key = str(st.secrets.get("OPENAI_API_KEY", ""))
    except Exception:
        secret_key = ""

    env_key = os.getenv("OPENAI_API_KEY", "")
    return (secret_key or env_key).strip()


def key_is_usable(api_key: str) -> bool:
    if not api_key:
        return False
    lowered = api_key.lower()
    if lowered in {"your_key_here", "sk-your-key-here", "paste_real_key_here"}:
        return False
    return api_key.startswith("sk-")


def _safe_error_msg(prefix: str, exc: Exception) -> str:
    return f"{prefix}. Open details for debug info."


@st.cache_resource(show_spinner=False)
def get_embeddings(api_key: str) -> OpenAIEmbeddings:
    return OpenAIEmbeddings(model=DEFAULT_EMBED_MODEL, api_key=api_key)


@st.cache_resource(show_spinner=False)
def get_llm(api_key: str) -> ChatOpenAI:
    return ChatOpenAI(model=DEFAULT_MODEL, temperature=0, api_key=api_key)


def vector_count(vectorstore: Chroma) -> int:
    try:
        return int(vectorstore._collection.count())  # noqa: SLF001
    except Exception:
        return 0



def extract_pdf_docs(file_bytes: bytes, source_name: str) -> List[Document]:
    reader = PdfReader(io.BytesIO(file_bytes))
    docs: List[Document] = []
    for idx, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        text = text.strip()
        if text:
            docs.append(
                Document(
                    page_content=text,
                    metadata={"source": source_name, "page": idx, "type": "pdf"},
                )
            )
    return docs



def extract_text_docs(file_bytes: bytes, source_name: str) -> List[Document]:
    text = file_bytes.decode("utf-8", errors="ignore").strip()
    if not text:
        return []
    return [
        Document(
            page_content=text,
            metadata={"source": source_name, "page": "-", "type": "text"},
        )
    ]



def parse_uploaded_files(uploaded_files) -> Tuple[List[Document], Dict[str, int]]:
    raw_docs: List[Document] = []
    stats = {"files": 0, "raw_docs": 0, "skipped_large": 0, "failed": 0}

    for up in uploaded_files:
        name = up.name
        b = up.read()
        if len(b) > MAX_UPLOAD_MB * 1024 * 1024:
            stats["skipped_large"] += 1
            continue
        ext = Path(name).suffix.lower()
        try:
            if ext == ".pdf":
                docs = extract_pdf_docs(b, name)
            else:
                docs = extract_text_docs(b, name)
        except Exception:
            stats["failed"] += 1
            continue

        raw_docs.extend(docs)
        stats["files"] += 1

    stats["raw_docs"] = len(raw_docs)
    return raw_docs, stats



def parse_local_paths(path_lines: str) -> Tuple[List[Document], Dict[str, int]]:
    raw_docs: List[Document] = []
    stats = {"files": 0, "raw_docs": 0, "missing": 0}
    for line in path_lines.splitlines():
        p = line.strip()
        if not p:
            continue
        fp = Path(p)
        if not fp.exists() or not fp.is_file():
            stats["missing"] += 1
            continue
        try:
            b = fp.read_bytes()
            if fp.suffix.lower() == ".pdf":
                docs = extract_pdf_docs(b, fp.name)
            else:
                docs = extract_text_docs(b, fp.name)
            raw_docs.extend(docs)
            stats["files"] += 1
        except Exception:
            stats["missing"] += 1

    stats["raw_docs"] = len(raw_docs)
    return raw_docs, stats



def chunk_documents(raw_docs: List[Document], chunk_size: int, chunk_overlap: int) -> List[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    split_docs = splitter.split_documents(raw_docs)
    for i, doc in enumerate(split_docs):
        doc.metadata["chunk_id"] = i
    return split_docs



def build_vector_db(split_docs: List[Document], api_key: str, rebuild: bool) -> Chroma:
    embeddings = get_embeddings(api_key)

    if rebuild and VECTOR_DB_DIR.exists():
        shutil.rmtree(VECTOR_DB_DIR, ignore_errors=True)

    VECTOR_DB_DIR.mkdir(parents=True, exist_ok=True)

    if split_docs:
        vectorstore = Chroma.from_documents(
            documents=split_docs,
            embedding=embeddings,
            persist_directory=str(VECTOR_DB_DIR),
            collection_name=COLLECTION_NAME,
        )
    else:
        vectorstore = Chroma(
            persist_directory=str(VECTOR_DB_DIR),
            collection_name=COLLECTION_NAME,
            embedding_function=embeddings,
        )

    # Refresh cache so subsequent reads use latest persisted collection.
    st.cache_resource.clear()
    return vectorstore



def load_vector_db(api_key: str) -> Chroma:
    embeddings = get_embeddings(api_key)
    return Chroma(
        persist_directory=str(VECTOR_DB_DIR),
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
    )



def build_prompt(prompt_mode: str) -> ChatPromptTemplate:
    if prompt_mode == "Reasoning Prompt":
        system_text = (
            "You are a document analyst. Use the retrieved context and chat history to answer. "
            "Reason step-by-step internally, but do not reveal private chain-of-thought. "
            "Return: (1) concise answer, (2) short rationale, (3) source citations."
        )
    else:
        system_text = (
            "You are a document analyst. Use only retrieved context and chat history. "
            "If uncertain, say not found in retrieved context. "
            "Return concise answer with source citations."
        )

    return ChatPromptTemplate.from_messages(
        [
            ("system", system_text),
            (
                "user",
                "Question:\n{question}\n\nChat History:\n{history}\n\nRetrieved Context:\n{context}",
            ),
        ]
    )



def rag_answer(
    vectorstore: Chroma,
    api_key: str,
    question: str,
    prompt_mode: str,
    history: List[Dict[str, str]],
    top_k: int,
) -> Tuple[str, List[Document]]:
    retriever = vectorstore.as_retriever(search_kwargs={"k": top_k})
    docs = retriever.invoke(question)

    context_blocks = []
    for i, d in enumerate(docs, start=1):
        source = d.metadata.get("source", "unknown")
        page = d.metadata.get("page", "-")
        chunk_id = d.metadata.get("chunk_id", "-")
        context_blocks.append(f"[{i}] {source} | page={page} | chunk={chunk_id}\n{d.page_content}")
    context = "\n\n".join(context_blocks)

    hist_txt = "\n".join([f"{m['role']}: {m['content']}" for m in history[-6:]]) or "No prior history"

    llm = get_llm(api_key)
    prompt = build_prompt(prompt_mode)
    chain = prompt | llm
    resp = chain.invoke({"question": question, "history": hist_txt, "context": context})

    return resp.content if hasattr(resp, "content") else str(resp), docs



def parse_eval_lines(raw_text: str) -> pd.DataFrame:
    rows = []
    for line in raw_text.splitlines():
        line = line.strip()
        if not line:
            continue
        if "||" not in line:
            continue
        q, a = line.split("||", 1)
        rows.append({"question": q.strip(), "ground_truth": a.strip()})
    return pd.DataFrame(rows)


if "doc_chat_history" not in st.session_state:
    st.session_state.doc_chat_history = []

with st.sidebar:
    st.header("Configuration")
    default_key = resolve_api_key()
    api_key = st.text_input("OpenAI API Key", value=default_key, type="password")
    prompt_mode = st.radio("Prompt Type", ["Single Prompt", "Reasoning Prompt"], index=0)
    chunk_size = st.slider("Chunk Size", min_value=300, max_value=2000, value=900, step=100)
    chunk_overlap = st.slider("Chunk Overlap", min_value=0, max_value=400, value=150, step=25)
    top_k = st.slider("Retriever Top-K", min_value=2, max_value=12, value=5)
    rebuild_db = st.checkbox("Rebuild Vector DB on ingest", value=False)
    if st.button("Clear Chat History"):
        st.session_state.doc_chat_history = []
        st.success("Chat history cleared.")
    st.caption("For sharing/deployment, set `OPENAI_API_KEY` in Streamlit Secrets.")
    st.caption("Vector DB path is local to the app runtime: `vector_db_docs/`.")
    st.caption(f"Max upload size per file: {MAX_UPLOAD_MB} MB")

ingest_tab, chat_tab, eval_tab = st.tabs(["Ingest + Vector DB", "RAG Chat", "QAEval Dashboard"])

with ingest_tab:
    st.subheader("1) Upload / Load Documents")
    uploaded = st.file_uploader(
        "Upload PDF, TXT, or textClipping files",
        type=["pdf", "txt", "textclipping"],
        accept_multiple_files=True,
    )
    use_local_paths = st.checkbox("Use local file paths (local machine only)", value=False)
    local_paths = st.text_area(
        "Optional local file paths (one per line)",
        value="",
        height=100,
        disabled=not use_local_paths,
        placeholder="/path/to/doc1.pdf\n/path/to/doc2.txt",
    )
    st.info("On hosted deployments, prefer file upload. Local absolute paths usually won't exist in cloud runtimes.")

    if st.button("Ingest, Chunk, and Build Vector DB"):
        if not key_is_usable(api_key):
            st.error("Provide a valid OPENAI_API_KEY (`sk-...`) first.")
        else:
            up_docs, up_stats = parse_uploaded_files(uploaded or [])
            path_docs, path_stats = parse_local_paths(local_paths) if use_local_paths else ([], {"files": 0, "raw_docs": 0, "missing": 0})
            raw_docs = up_docs + path_docs

            if not raw_docs:
                st.error("No readable documents found from uploads/paths.")
            else:
                chunks = chunk_documents(raw_docs, chunk_size, chunk_overlap)
                build_vector_db(chunks, api_key, rebuild=rebuild_db)

                st.success("Vector DB updated successfully.")
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Uploaded Files", up_stats["files"])
                c2.metric("Path Files", path_stats["files"])
                c3.metric("Raw Docs", len(raw_docs))
                c4.metric("Chunks", len(chunks))
                if path_stats.get("missing", 0) > 0:
                    st.warning(f"Skipped {path_stats['missing']} missing/unreadable local paths.")
                if up_stats.get("skipped_large", 0) > 0:
                    st.warning(f"Skipped {up_stats['skipped_large']} oversized uploads (> {MAX_UPLOAD_MB} MB each).")
                if up_stats.get("failed", 0) > 0:
                    st.warning(f"Failed to parse {up_stats['failed']} uploaded files.")

with chat_tab:
    st.subheader("2) RAG Interaction UI")
    if not key_is_usable(api_key):
        st.warning("Set a valid `OPENAI_API_KEY` (`sk-...`) to run retrieval + LLM responses.")

    for m in st.session_state.doc_chat_history:
        with st.chat_message(m["role"]):
            st.markdown(m["content"])

    q = st.chat_input("Ask questions about uploaded documents")
    if q:
        st.session_state.doc_chat_history.append({"role": "user", "content": q})
        with st.chat_message("user"):
            st.markdown(q)

        try:
            vs = load_vector_db(api_key)
            if vector_count(vs) == 0:
                raise ValueError("Vector DB is empty. Ingest documents first.")
            answer, docs = rag_answer(vs, api_key, q, prompt_mode, st.session_state.doc_chat_history, top_k)
            st.session_state.doc_chat_history.append({"role": "assistant", "content": answer})

            with st.chat_message("assistant"):
                st.markdown(answer)
                with st.expander("Retrieved Chunks"):
                    for i, d in enumerate(docs, start=1):
                        st.markdown(
                            f"**[{i}] {d.metadata.get('source','unknown')} | page={d.metadata.get('page','-')} | chunk={d.metadata.get('chunk_id','-')}**"
                        )
                        st.write(d.page_content[:900] + ("..." if len(d.page_content) > 900 else ""))
        except Exception as exc:
            msg = _safe_error_msg("RAG call failed", exc)
            st.session_state.doc_chat_history.append({"role": "assistant", "content": msg})
            with st.chat_message("assistant"):
                st.error(msg)
                with st.expander("Debug Details"):
                    st.code(str(exc))

with eval_tab:
    st.subheader("3) QAEvalChain Evaluation + Metrics")
    st.caption("Format: question || ground_truth_answer (one pair per line)")
    eval_default = (
        "Which product performed best in the latest quarter? || Widget D performed best in the latest quarter based on retrieved context.\n"
        "What is the main focus of AI business model innovation paper? || The paper focuses on applying AI to create or transform business model components.\n"
        "What does the Walmart sales analysis document discuss? || It discusses Walmart sales trends and analysis using business intelligence methods."
    )
    eval_lines = st.text_area("Evaluation Set", value=eval_default, height=180)

    if st.button("Run QAEvalChain"):
        if not key_is_usable(api_key):
            st.error("Provide a valid OPENAI_API_KEY (`sk-...`) first.")
        else:
            eval_df = parse_eval_lines(eval_lines)
            if eval_df.empty:
                st.error("No valid evaluation rows found. Use `question || answer` format.")
            else:
                try:
                    vs = load_vector_db(api_key)
                    if vector_count(vs) == 0:
                        raise ValueError("Vector DB is empty. Ingest documents first.")
                    preds = []
                    for q in eval_df["question"]:
                        pred, _ = rag_answer(vs, api_key, q, prompt_mode, history=[], top_k=top_k)
                        preds.append(pred)
                    eval_df["prediction"] = preds

                    llm = get_llm(api_key)
                    qa_chain = QAEvalChain.from_llm(llm)
                    examples = [{"query": r.question, "answer": r.ground_truth} for r in eval_df.itertuples(index=False)]
                    predictions = [{"result": r.prediction} for r in eval_df.itertuples(index=False)]

                    graded = qa_chain.evaluate(
                        examples=examples,
                        predictions=predictions,
                        question_key="query",
                        answer_key="answer",
                        prediction_key="result",
                    )

                    grades = []
                    reasons = []
                    for g in graded:
                        txt = (g.get("text") or str(g)).strip()
                        up = txt.upper()
                        label = "INCORRECT"
                        if "CORRECT" in up and "INCORRECT" not in up:
                            label = "CORRECT"
                        grades.append(label)
                        reasons.append(txt)

                    eval_df["qaeval_grade"] = grades
                    eval_df["qaeval_reason"] = reasons

                    correct = int((eval_df["qaeval_grade"] == "CORRECT").sum())
                    total = len(eval_df)
                    incorrect = total - correct
                    accuracy = (correct / total * 100) if total else 0.0

                    m1, m2, m3 = st.columns(3)
                    m1.metric("Accuracy", f"{accuracy:.2f}%")
                    m2.metric("Correct", correct)
                    m3.metric("Incorrect", incorrect)

                    chart_df = pd.DataFrame({"grade": ["CORRECT", "INCORRECT"], "count": [correct, incorrect]})
                    st.bar_chart(chart_df.set_index("grade"))

                    st.dataframe(eval_df, use_container_width=True)

                    out_path = Path("analysis_output/doc_qaeval_results.csv")
                    out_path.parent.mkdir(parents=True, exist_ok=True)
                    eval_df.to_csv(out_path, index=False)
                    st.success(f"Saved results to {out_path}")
                    st.download_button(
                        "Download QAEval Results CSV",
                        data=eval_df.to_csv(index=False).encode("utf-8"),
                        file_name="doc_qaeval_results.csv",
                        mime="text/csv",
                    )
                except Exception as exc:
                    st.error(_safe_error_msg("QAEval failed", exc))
                    with st.expander("Debug Details"):
                        st.code(str(exc))
