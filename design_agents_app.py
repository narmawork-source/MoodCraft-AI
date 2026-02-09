import os
import time
import uuid
import base64
import json
from collections import defaultdict
from datetime import datetime
from io import BytesIO

import pandas as pd
import streamlit as st
from PIL import Image
from langchain_chroma import Chroma
from langchain_classic.evaluation.qa import QAEvalChain
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from openai import OpenAI

from design_agents.moodboard_agent import compose_moodboard
from design_agents.retail_agent import search_products
from design_agents.style_agent import analyze_style_dna

st.set_page_config(page_title="MoodCraft AI", page_icon="🛋️", layout="wide")

st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;700&display=swap');
html, body, [class*="css"] { font-family: 'Space Grotesk', sans-serif; }
:root {
  --ink: #1b1f2a;
  --warm: #d67a3d;
  --stone: #f8f3ea;
  --leaf: #2f6f63;
  --sun: #f2b35f;
  --mist: #e9eef7;
}
.stApp {
  background:
    radial-gradient(1200px 500px at 95% -10%, #dbe9ff 0%, rgba(219,233,255,0) 60%),
    radial-gradient(900px 450px at -15% 12%, #ffe4c8 0%, rgba(255,228,200,0) 58%),
    linear-gradient(180deg, #fbfaf7 0%, #f7f3ec 100%);
}
.block-container { padding-top: 1.2rem; max-width: 1200px; }
.mc-hero {
  background:
    linear-gradient(135deg, rgba(30,40,70,.92) 0%, rgba(45,84,96,.92) 45%, rgba(214,122,61,.88) 100%),
    url('https://images.unsplash.com/photo-1616594039964-3dd9f9f142c0?auto=format&fit=crop&w=1200&q=60');
  background-size: cover;
  border-radius: 24px;
  padding: 1.6rem 1.5rem;
  margin-bottom: 1rem;
  box-shadow: 0 12px 30px rgba(21,31,58,.18);
}
.mc-badge {
  display: inline-block;
  background: rgba(255,255,255,.2);
  border: 1px solid rgba(255,255,255,.4);
  color: #ffffff;
  border-radius: 999px;
  padding: .2rem .7rem;
  font-size: .78rem;
  margin-bottom: .5rem;
}
.mc-title { color: #ffffff; font-size: 2.2rem; font-weight: 700; margin: 0; letter-spacing: .2px; }
.mc-sub { color: #e7eefb; margin-top: .35rem; max-width: 700px; }
.mc-kpis {
  display: grid;
  grid-template-columns: repeat(4, minmax(0, 1fr));
  gap: .55rem;
  margin-top: 1rem;
}
.mc-kpi {
  background: rgba(255,255,255,.16);
  border: 1px solid rgba(255,255,255,.36);
  border-radius: 12px;
  color: #fff;
  padding: .55rem .65rem;
}
.mc-kpi .v { font-size: 1.05rem; font-weight: 700; line-height: 1.2; }
.mc-kpi .l { font-size: .74rem; opacity: .95; }
.mc-front-grid {
  display: grid;
  grid-template-columns: repeat(3, minmax(0, 1fr));
  gap: .7rem;
  margin-bottom: .9rem;
}
.mc-front-card {
  background: #ffffff;
  border: 1px solid #e7ddcf;
  border-radius: 14px;
  padding: .85rem .95rem;
  box-shadow: 0 6px 16px rgba(40,48,67,.06);
}
.mc-front-card h4 {
  margin: .1rem 0 .35rem 0;
  color: var(--ink);
  font-size: 1rem;
}
.mc-front-card p {
  margin: 0;
  color: #485364;
  font-size: .86rem;
  line-height: 1.35;
}
.mc-agent-head {
  display: flex;
  align-items: center;
  gap: .6rem;
  margin-bottom: .35rem;
}
.mc-avatar {
  width: 44px;
  height: 44px;
  border-radius: 999px;
  border: 1px solid #e3d8c9;
  object-fit: cover;
}
.mc-card {
  background: #ffffff;
  border: 1px solid #e8dfd3;
  border-radius: 14px;
  padding: .8rem 1rem;
}
.mc-pill {
  display: inline-block;
  background: #f1ebe1;
  color: #4e4030;
  border-radius: 999px;
  padding: .15rem .6rem;
  margin: .15rem .2rem .15rem 0;
  font-size: .82rem;
}
.mc-step {
  background: #faf8f4;
  border-left: 4px solid var(--warm);
  border-radius: 8px;
  padding: .55rem .75rem;
  margin-bottom: .5rem;
}
@media (max-width: 900px) {
  .mc-kpis { grid-template-columns: repeat(2, minmax(0, 1fr)); }
  .mc-front-grid { grid-template-columns: 1fr; }
}
</style>
""",
    unsafe_allow_html=True,
)

st.markdown(
    """
<div class="mc-hero">
  <span class="mc-badge">Canva-Style Template</span>
  <p class="mc-title">MoodCraft AI</p>
  <p class="mc-sub">Three-agent interior design copilot that analyzes your room, sources products, composes moodboards, and measures quality with QAEval.</p>
  <div class="mc-kpis">
    <div class="mc-kpi"><div class="v">3 Agents</div><div class="l">Style, Retail, Moodboard</div></div>
    <div class="mc-kpi"><div class="v">Image KB</div><div class="l">Chunked + Vector Search</div></div>
    <div class="mc-kpi"><div class="v">QAEval</div><div class="l">Prompt-Level Evaluation</div></div>
    <div class="mc-kpi"><div class="v">Feedback</div><div class="l">Thumbs Up / Down Metrics</div></div>
  </div>
</div>
<div class="mc-front-grid">
  <div class="mc-front-card">
    <h4>1) Upload + Interpret</h4>
    <p>Upload room and inspiration images, then extract style DNA: tags, palette, materials, and constraints.</p>
  </div>
  <div class="mc-front-card">
    <h4>2) Source + Compose</h4>
    <p>Find matching products by budget and generate a coherent moodboard with placement guidance.</p>
  </div>
  <div class="mc-front-card">
    <h4>3) Evaluate + Improve</h4>
    <p>Track answer quality by prompt, rank agent performance, and tune models/creativity in real time.</p>
  </div>
</div>
""",
    unsafe_allow_html=True,
)

if "style_dna" not in st.session_state:
    st.session_state.style_dna = {}
if "products" not in st.session_state:
    st.session_state.products = []
if "board" not in st.session_state:
    st.session_state.board = {}
if "generated_board_image" not in st.session_state:
    st.session_state.generated_board_image = None
if "qa_log" not in st.session_state:
    st.session_state.qa_log = []
if "mcp_trace" not in st.session_state:
    st.session_state.mcp_trace = []
if "image_ingest_log" not in st.session_state:
    st.session_state.image_ingest_log = []
if "agent_chat_log" not in st.session_state:
    st.session_state.agent_chat_log = []
if "feedback_log" not in st.session_state:
    st.session_state.feedback_log = []
if "last_qa_answer" not in st.session_state:
    st.session_state.last_qa_answer = ""
if "last_qa_question" not in st.session_state:
    st.session_state.last_qa_question = ""
if "last_image_answer" not in st.session_state:
    st.session_state.last_image_answer = ""
if "last_image_question" not in st.session_state:
    st.session_state.last_image_question = ""
if "agent_eval_log" not in st.session_state:
    st.session_state.agent_eval_log = []

IMAGE_VDB_DIR = "moodcraft_image_vdb"
IMAGE_COLLECTION = "moodcraft_image_chunks"


@st.cache_data(show_spinner=False)
def load_avatar_data_uri(path: str) -> str:
    with open(path, "rb") as f:
        b = base64.b64encode(f.read()).decode("utf-8")
    return f"data:image/svg+xml;base64,{b}"


def get_image_vector_store(api_key: str) -> Chroma:
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small", api_key=api_key)
    return Chroma(
        collection_name=IMAGE_COLLECTION,
        persist_directory=IMAGE_VDB_DIR,
        embedding_function=embeddings,
    )


def tile_descriptor(tile: Image.Image, image_name: str, tile_index: int, prompt_hint: str) -> str:
    q = tile.convert("RGB").resize((64, 64)).quantize(colors=3)
    pal = q.getpalette()[:9]
    colors = []
    for i in range(0, len(pal), 3):
        if i + 2 < len(pal):
            colors.append(f"#{pal[i]:02x}{pal[i+1]:02x}{pal[i+2]:02x}")
    stat = tile.convert("L").resize((32, 32))
    mean_luma = int(sum(stat.getdata()) / (32 * 32))
    vibe = "bright" if mean_luma > 150 else "moody" if mean_luma < 90 else "balanced"
    return (
        f"Image {image_name}, tile {tile_index}, dominant colors {', '.join(colors[:3])}, "
        f"lighting {vibe}, user style prompt {prompt_hint}"
    )


def chunk_image_to_documents(image_bytes: bytes, image_name: str, prompt_hint: str, grid: int = 3) -> list[Document]:
    img = Image.open(BytesIO(image_bytes)).convert("RGB")
    img = img.resize((600, 600))
    w, h = img.size
    tw, th = w // grid, h // grid
    docs: list[Document] = []
    idx = 0
    for r in range(grid):
        for c in range(grid):
            box = (c * tw, r * th, (c + 1) * tw, (r + 1) * th)
            tile = img.crop(box)
            desc = tile_descriptor(tile, image_name, idx, prompt_hint)
            docs.append(
                Document(
                    page_content=desc,
                    metadata={"image_name": image_name, "tile_index": idx, "grid": grid},
                )
            )
            idx += 1
    return docs


def retrieve_image_matches(vs: Chroma, query: str, top_k: int = 8) -> list[dict]:
    docs_scores = vs.similarity_search_with_score(query, k=top_k)
    agg = defaultdict(list)
    for d, score in docs_scores:
        img = d.metadata.get("image_name", "unknown")
        agg[img].append(float(score))
    out = []
    for img, scores in agg.items():
        avg_dist = sum(scores) / len(scores)
        match = round(1.0 / (1.0 + avg_dist), 3)
        out.append({"image_name": img, "match_score": match, "avg_distance": round(avg_dist, 3)})
    out.sort(key=lambda x: x["match_score"], reverse=True)
    return out


def run_live_web_search(api_key: str, question: str, llm_model: str) -> str:
    client = OpenAI(api_key=api_key)
    web_prompt = (
        "Use live web results to answer the user's design-shopping question. "
        "Return concise bullets with source links when possible."
    )
    try:
        resp = client.responses.create(
            model=llm_model,
            input=f"{web_prompt}\n\nQuestion: {question}",
            tools=[{"type": "web_search_preview"}],
        )
        out = getattr(resp, "output_text", "")
        if out:
            return out
    except Exception:
        pass

    try:
        resp = client.responses.create(
            model=llm_model,
            input=f"{web_prompt}\n\nQuestion: {question}",
            tools=[{"type": "web_search"}],
        )
        return getattr(resp, "output_text", "") or "No live web response returned."
    except Exception as exc:
        return f"Live web search unavailable: {exc}"


def generate_moodboard_image(
    api_key: str,
    design_type: str,
    style_dna: dict,
    products: list[dict],
    image_model: str,
    creativity: float,
) -> bytes:
    client = OpenAI(api_key=api_key)
    top_items = ", ".join([p.get("title", "item") for p in products[:5]])
    palette = ", ".join([p.get("hex", "") for p in style_dna.get("palette", [])[:5]])
    mats = ", ".join(style_dna.get("materials", [])[:5])
    prompt = (
        "Create a polished interior design moodboard image with product collage layout. "
        f"Design type: {design_type}. "
        f"Style tags: {', '.join(style_dna.get('style_tags', []))}. "
        f"Palette: {palette}. Materials: {mats}. "
        f"Include visual cues for these items: {top_items}. "
        "Keep composition clean, premium, and presentation-ready."
    )
    if creativity >= 0.65:
        prompt += " Explore bolder composition, richer contrast, and more expressive styling details."
    elif creativity <= 0.25:
        prompt += " Keep the layout highly practical, minimal, and conservative."
    resp = client.images.generate(
        model=image_model,
        prompt=prompt,
        size="1024x1024",
    )
    b64 = resp.data[0].b64_json
    return base64.b64decode(b64)


def trace(tool: str, ok: bool, ms: int, err: str = ""):
    st.session_state.mcp_trace.append(
        {
            "timestamp": datetime.utcnow().isoformat(timespec="seconds") + "Z",
            "tool": tool,
            "status": "success" if ok else "error",
            "latency_ms": ms,
            "error": err,
        }
    )


def eval_answer(api_key: str, question: str, reference: str, prediction: str) -> dict:
    llm = ChatOpenAI(model="gpt-4.1-mini", temperature=0, api_key=api_key)
    chain = QAEvalChain.from_llm(llm)
    out = chain.evaluate(
        examples=[{"query": question, "answer": reference}],
        predictions=[{"result": prediction}],
        question_key="query",
        answer_key="answer",
        prediction_key="result",
    )
    txt = (out[0].get("text") or str(out[0])).strip()
    up = txt.upper()
    grade = "INCORRECT"
    if "CORRECT" in up and "INCORRECT" not in up:
        grade = "CORRECT"
    return {"grade": grade, "reason": txt}


def evaluate_agents(api_key: str, llm_model: str, question: str, style_dna: dict, products: list, board: dict, final_answer: str) -> dict:
    llm = ChatOpenAI(model=llm_model, temperature=0, api_key=api_key)
    prompt = (
        "Score each agent from 1 to 10 for how useful they are for answering the user question.\n"
        "Agents: style_agent, retail_agent, moodboard_agent, orchestrator.\n"
        "Return strict JSON only with keys: style_agent, retail_agent, moodboard_agent, orchestrator, rationale.\n"
        "Each agent key must contain: score (int), reason (string).\n\n"
        f"Question: {question}\n"
        f"Style DNA: {style_dna}\n"
        f"Products: {products[:8]}\n"
        f"Moodboard: {board}\n"
        f"Final Answer: {final_answer}"
    )
    try:
        raw = llm.invoke(prompt).content
        data = json.loads(raw)
    except Exception:
        data = {
            "style_agent": {"score": 0, "reason": "Evaluation parse failed."},
            "retail_agent": {"score": 0, "reason": "Evaluation parse failed."},
            "moodboard_agent": {"score": 0, "reason": "Evaluation parse failed."},
            "orchestrator": {"score": 0, "reason": "Evaluation parse failed."},
            "rationale": "Fallback due to JSON parsing error.",
        }
    return data


def render_style_dna(dna: dict):
    st.markdown('<div class="mc-card">', unsafe_allow_html=True)
    st.markdown("**Style Tags**")
    st.markdown("".join([f'<span class="mc-pill">{t}</span>' for t in dna.get("style_tags", [])]), unsafe_allow_html=True)

    st.markdown("**Materials**")
    st.markdown("".join([f'<span class="mc-pill">{t}</span>' for t in dna.get("materials", [])]), unsafe_allow_html=True)

    st.markdown("**Palette**")
    palette = dna.get("palette", [])
    if palette:
        cols = st.columns(min(6, len(palette)))
        for i, p in enumerate(palette[:6]):
            h = p.get("hex", "#cccccc")
            u = p.get("usage", "")
            cols[i].markdown(
                f"<div style='background:{h};height:48px;border-radius:10px;border:1px solid #ddd;'></div>"
                f"<div style='font-size:.8rem'>{h}<br>{u}</div>",
                unsafe_allow_html=True,
            )

    st.markdown("**Do / Don't**")
    c1, c2 = st.columns(2)
    c1.write(dna.get("do_constraints", []))
    c2.write(dna.get("dont_constraints", []))
    st.markdown("</div>", unsafe_allow_html=True)


def render_products(products: list):
    cols = st.columns(3)
    for i, p in enumerate(products):
        with cols[i % 3]:
            st.markdown('<div class="mc-card">', unsafe_allow_html=True)
            st.write(f"**{p.get('title','Item')}**")
            st.write(f"Retailer: {p.get('retailer','N/A')}")
            st.write(f"Price: ${p.get('price','N/A')}")
            st.write(f"Match: {p.get('match_score',0)}")
            st.markdown(f"[Open Product]({p.get('url','#')})")
            st.markdown("</div>", unsafe_allow_html=True)


def render_board(board: dict):
    st.markdown('<div class="mc-card">', unsafe_allow_html=True)
    st.write(board.get("design_story", ""))
    items = board.get("board_items", [])
    if items:
        st.dataframe(pd.DataFrame(items), use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)


def render_feedback_controls(scope: str, question: str, answer: str):
    st.markdown("**Feedback**")
    c_up, c_down = st.columns(2)
    up_key = f"{scope}_up_{abs(hash(question + answer))}"
    down_key = f"{scope}_down_{abs(hash(question + answer))}"
    with c_up:
        if st.button("👍 Thumbs Up", key=up_key, use_container_width=True):
            st.session_state.feedback_log.append(
                {
                    "timestamp": datetime.utcnow().isoformat(timespec="seconds") + "Z",
                    "scope": scope,
                    "question": question,
                    "answer": answer[:500],
                    "feedback": "up",
                }
            )
            st.success("Thanks for the feedback.")
    with c_down:
        if st.button("👎 Thumbs Down", key=down_key, use_container_width=True):
            st.session_state.feedback_log.append(
                {
                    "timestamp": datetime.utcnow().isoformat(timespec="seconds") + "Z",
                    "scope": scope,
                    "question": question,
                    "answer": answer[:500],
                    "feedback": "down",
                }
            )
            st.success("Thanks for the feedback.")


with st.sidebar:
    st.subheader("Control Center")
    api_key = st.text_input("OpenAI API Key", value=os.getenv("OPENAI_API_KEY", ""), type="password")
    llm_model = st.selectbox(
        "LLM Model",
        ["gpt-4.1-mini", "gpt-4.1", "gpt-4o-mini"],
        index=0,
    )
    image_model = st.selectbox(
        "Image Model",
        ["gpt-image-1", "dall-e-3"],
        index=0,
    )
    creativity_enabled = st.toggle("Creativity Mode", value=False)
    creativity_level = st.slider(
        "Creativity Level",
        0.0,
        1.0,
        0.7 if creativity_enabled else 0.0,
        0.05,
        disabled=not creativity_enabled,
    )
    design_type = st.selectbox(
        "Design Type",
        [
            "Modern",
            "Industrial",
            "Contemporary",
            "Traditional",
        ],
        index=0,
    )
    budget_min, budget_max = st.slider("Budget Range", 20, 1500, (100, 400), step=10)
    decor_text = st.text_input("Decor Types (comma)", value="rug,lamp,console")
    auto_qaeval = st.checkbox("Auto QAEval", value=True)

    st.markdown("---")
    st.markdown("**Workflow**")
    st.markdown('<div class="mc-step">1. Run Agent 1 to generate design DNA.</div>', unsafe_allow_html=True)
    st.markdown('<div class="mc-step">2. Run Agent 2 to source matching products.</div>', unsafe_allow_html=True)
    st.markdown('<div class="mc-step">3. Run Agent 3 to compose moodboard.</div>', unsafe_allow_html=True)
    st.markdown('<div class="mc-step">4. Ask questions and evaluate quality.</div>', unsafe_allow_html=True)


style_avatar_uri = load_avatar_data_uri("design_agents/assets/style_avatar.svg")
retail_avatar_uri = load_avatar_data_uri("design_agents/assets/retail_avatar.svg")
board_avatar_uri = load_avatar_data_uri("design_agents/assets/board_avatar.svg")
orchestrator_avatar_uri = load_avatar_data_uri("design_agents/assets/orchestrator_avatar.svg")


t0, t1, t2, t3, t4, t5 = st.tabs(
    ["Meet Our Agents", "Agent 1: Style", "Agent 2: Retail", "Agent 3: Moodboard", "QAEval + Trace", "Image KB + Agent Chat"]
)

with t0:
    st.subheader("Meet Our Agents")
    st.caption("Each agent has a focused role and they collaborate to produce a complete interior design recommendation.")

    c1, c2 = st.columns(2)
    with c1:
        st.markdown('<div class="mc-card">', unsafe_allow_html=True)
        st.markdown(
            f"""
            <div class="mc-agent-head">
              <img class="mc-avatar" src="{style_avatar_uri}" alt="Style Agent Avatar" />
              <h3 style="margin:0">Agent 1: Style Interpreter</h3>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.write("**Persona:** Visual Design Strategist")
        st.write("**Mission:** Understand room photos and user intent to build structured design DNA.")
        st.write("**What it does:**")
        st.write("- Extracts style tags (modern, industrial, traditional, etc.)")
        st.write("- Builds palette and material direction")
        st.write("- Identifies decor types and do/don't constraints")
        st.markdown("</div>", unsafe_allow_html=True)

    with c2:
        st.markdown('<div class="mc-card">', unsafe_allow_html=True)
        st.markdown(
            f"""
            <div class="mc-agent-head">
              <img class="mc-avatar" src="{retail_avatar_uri}" alt="Retail Agent Avatar" />
              <h3 style="margin:0">Agent 2: Retail Sourcing</h3>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.write("**Persona:** Budget-Conscious Product Curator")
        st.write("**Mission:** Convert style DNA into practical, purchasable product recommendations.")
        st.write("**What it does:**")
        st.write("- Selects products by decor type and budget")
        st.write("- Prioritizes style fit and cost")
        st.write("- Returns ranked product options with links")
        st.markdown("</div>", unsafe_allow_html=True)

    c3, c4 = st.columns(2)
    with c3:
        st.markdown('<div class="mc-card">', unsafe_allow_html=True)
        st.markdown(
            f"""
            <div class="mc-agent-head">
              <img class="mc-avatar" src="{board_avatar_uri}" alt="Moodboard Agent Avatar" />
              <h3 style="margin:0">Agent 3: Moodboard Composer</h3>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.write("**Persona:** Spatial Storyteller")
        st.write("**Mission:** Synthesize products + style into a coherent visual and design narrative.")
        st.write("**What it does:**")
        st.write("- Composes moodboard item set")
        st.write("- Produces design story and placement guidance")
        st.write("- Supports AI image render generation")
        st.markdown("</div>", unsafe_allow_html=True)

    with c4:
        st.markdown('<div class="mc-card">', unsafe_allow_html=True)
        st.markdown(
            f"""
            <div class="mc-agent-head">
              <img class="mc-avatar" src="{orchestrator_avatar_uri}" alt="Orchestrator Avatar" />
              <h3 style="margin:0">Orchestrator + Evaluator</h3>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.write("**Persona:** Quality Lead")
        st.write("**Mission:** Coordinate agent outputs and validate final answer quality.")
        st.write("**What it does:**")
        st.write("- Combines retrieved context + agent outputs")
        st.write("- Generates final response to user prompts")
        st.write("- Runs QAEval, ranking, and feedback analytics")
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("**How they collaborate**")
    st.write("1. Style Interpreter creates design DNA from prompt/image context.")
    st.write("2. Retail Sourcing finds products that match style + budget.")
    st.write("3. Moodboard Composer builds a cohesive concept and presentation.")
    st.write("4. Orchestrator answers user questions and tracks quality with QAEval + feedback.")

with t1:
    st.subheader("Style Interpreter")
    prompt = st.text_input("Design Prompt", value=f"cozy {design_type.lower()} living room")
    if st.button("Run Style Interpreter", use_container_width=True):
        started = time.perf_counter()
        dna = analyze_style_dna(f"{design_type.lower()} | {prompt}")
        elapsed = int((time.perf_counter() - started) * 1000)
        trace("analyze_room_images", True, elapsed)
        st.session_state.style_dna = dna

    if st.session_state.style_dna:
        render_style_dna(st.session_state.style_dna)

with t2:
    st.subheader("Retail Sourcing")
    if st.button("Run Retail Sourcing", use_container_width=True):
        dna = st.session_state.style_dna
        if not dna:
            st.error("Run Agent 1 first.")
        else:
            decor = [x.strip() for x in decor_text.split(",") if x.strip()]
            started = time.perf_counter()
            products = search_products(dna, decor, budget_min, budget_max)
            elapsed = int((time.perf_counter() - started) * 1000)
            trace("search_products", True, elapsed)
            st.session_state.products = products

    if st.session_state.products:
        render_products(st.session_state.products)

with t3:
    st.subheader("Moodboard Composer")
    if st.button("Run Moodboard Composer", use_container_width=True):
        dna = st.session_state.style_dna
        products = st.session_state.products
        if not dna or not products:
            st.error("Run Agent 1 and 2 first.")
        else:
            started = time.perf_counter()
            board = compose_moodboard(dna, products)
            elapsed = int((time.perf_counter() - started) * 1000)
            trace("compose_moodboard", True, elapsed)
            st.session_state.board = board

    if st.session_state.board:
        render_board(st.session_state.board)

    st.markdown("---")
    st.markdown(f"**AI Moodboard Render ({image_model})**")
    if st.button("Generate Moodboard Image", use_container_width=True):
        if not api_key:
            st.error("Set OPENAI_API_KEY.")
        elif not st.session_state.style_dna or not st.session_state.products:
            st.error("Run Agent 1 and Agent 2 first.")
        else:
            try:
                img_bytes = generate_moodboard_image(
                    api_key=api_key,
                    design_type=design_type,
                    style_dna=st.session_state.style_dna,
                    products=st.session_state.products,
                    image_model=image_model,
                    creativity=creativity_level,
                )
                st.session_state.generated_board_image = img_bytes
                st.success("Moodboard image generated.")
            except Exception as exc:
                st.error(f"Image generation failed: {exc}")

    if st.session_state.generated_board_image:
        st.image(st.session_state.generated_board_image, caption="Generated Moodboard", use_container_width=True)
        st.download_button(
            "Download Moodboard PNG",
            data=st.session_state.generated_board_image,
            file_name="moodcraft_board.png",
            mime="image/png",
        )

with t4:
    st.subheader("Q&A + Automated QAEval")
    question = st.text_input("Ask a design question")
    if st.button("Ask", use_container_width=True):
        if not api_key:
            st.error("Set OPENAI_API_KEY.")
        else:
            dna = st.session_state.style_dna
            products = st.session_state.products
            board = st.session_state.board
            llm = ChatOpenAI(model=llm_model, temperature=creativity_level, api_key=api_key)
            answer = llm.invoke(
                f"Question: {question}\nStyle:{dna}\nProducts:{products[:5]}\nBoard:{board}"
            ).content
            st.session_state.last_qa_answer = answer
            st.session_state.last_qa_question = question
            st.markdown("**Answer**")
            st.write(answer)

            if auto_qaeval:
                reference = "Reference should align with style, products, and moodboard context."
                ev = eval_answer(api_key, question, reference, answer)
                agent_eval = evaluate_agents(
                    api_key=api_key,
                    llm_model=llm_model,
                    question=question,
                    style_dna=dna,
                    products=products,
                    board=board,
                    final_answer=answer,
                )
                st.caption(f"Auto QAEval: {ev['grade']}")
                st.session_state.qa_log.append(
                    {
                        "question": question,
                        "reference": reference,
                        "prediction": answer,
                        "qaeval_grade": ev["grade"],
                        "qaeval_reason": ev["reason"],
                        "style_agent_score": agent_eval.get("style_agent", {}).get("score", 0),
                        "retail_agent_score": agent_eval.get("retail_agent", {}).get("score", 0),
                        "moodboard_agent_score": agent_eval.get("moodboard_agent", {}).get("score", 0),
                        "orchestrator_score": agent_eval.get("orchestrator", {}).get("score", 0),
                    }
                )
                st.session_state.agent_eval_log.append(
                    {
                        "timestamp": datetime.utcnow().isoformat(timespec="seconds") + "Z",
                        "question": question,
                        "style_agent_score": agent_eval.get("style_agent", {}).get("score", 0),
                        "style_agent_reason": agent_eval.get("style_agent", {}).get("reason", ""),
                        "retail_agent_score": agent_eval.get("retail_agent", {}).get("score", 0),
                        "retail_agent_reason": agent_eval.get("retail_agent", {}).get("reason", ""),
                        "moodboard_agent_score": agent_eval.get("moodboard_agent", {}).get("score", 0),
                        "moodboard_agent_reason": agent_eval.get("moodboard_agent", {}).get("reason", ""),
                        "orchestrator_score": agent_eval.get("orchestrator", {}).get("score", 0),
                        "orchestrator_reason": agent_eval.get("orchestrator", {}).get("reason", ""),
                    }
                )

    if st.session_state.last_qa_answer:
        render_feedback_controls(
            scope="qa_tab",
            question=st.session_state.last_qa_question,
            answer=st.session_state.last_qa_answer,
        )

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**QAEval Log**")
        if st.session_state.qa_log:
            df = pd.DataFrame(st.session_state.qa_log)
            total = len(df)
            correct = int((df["qaeval_grade"] == "CORRECT").sum())
            st.metric("QAEval Accuracy", f"{(correct / total * 100):.2f}%")
            if {"style_agent_score", "retail_agent_score", "moodboard_agent_score", "orchestrator_score"}.issubset(df.columns):
                rank_rows = [
                    {"agent": "orchestrator", "avg_score": float(df["orchestrator_score"].mean())},
                    {"agent": "style_agent", "avg_score": float(df["style_agent_score"].mean())},
                    {"agent": "retail_agent", "avg_score": float(df["retail_agent_score"].mean())},
                    {"agent": "moodboard_agent", "avg_score": float(df["moodboard_agent_score"].mean())},
                ]
                rank_df = pd.DataFrame(rank_rows).sort_values("avg_score", ascending=False).reset_index(drop=True)
                rank_df.index = rank_df.index + 1
                rank_df.index.name = "rank"
                st.markdown("**Agent Ranking (Avg Score /10)**")
                st.dataframe(rank_df, use_container_width=True)
            st.dataframe(df, use_container_width=True)
    with c2:
        st.markdown("**MCP Trace Panel**")
        if st.session_state.mcp_trace:
            tr = pd.DataFrame(st.session_state.mcp_trace)
            st.dataframe(tr, use_container_width=True)
        st.markdown("**Feedback Metrics**")
        if st.session_state.feedback_log:
            fdf = pd.DataFrame(st.session_state.feedback_log)
            up = int((fdf["feedback"] == "up").sum())
            down = int((fdf["feedback"] == "down").sum())
            st.metric("Thumbs Up", up)
            st.metric("Thumbs Down", down)
            by_scope = (
                fdf.groupby(["scope", "feedback"])
                .size()
                .reset_index(name="count")
                .sort_values(["scope", "feedback"])
            )
            st.dataframe(by_scope, use_container_width=True)

with t5:
    st.subheader("Image Upload + Vector Search + Agent Conversation")
    st.caption("Upload images, chunk into tiles, store in vector DB, then chat to match image vibe and generate moodboard.")
    enable_live_web = st.checkbox("Enable Live Web Search in this tab", value=False)

    image_prompt = st.text_input("Image Intent Prompt", value="warm modern living room with natural textures")
    uploaded_images = st.file_uploader(
        "Upload room/inspiration images",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=True,
        key="image_kb_uploader",
    )

    c_ing1, c_ing2 = st.columns([2, 1])
    with c_ing1:
        if st.button("Chunk + Store Images in Vector DB", use_container_width=True):
            if not api_key:
                st.error("Set OPENAI_API_KEY first.")
            elif not uploaded_images:
                st.error("Upload at least one image.")
            else:
                try:
                    vs = get_image_vector_store(api_key)
                    total_chunks = 0
                    for f in uploaded_images:
                        docs = chunk_image_to_documents(f.getvalue(), f.name, image_prompt, grid=3)
                        ids = [str(uuid.uuid4()) for _ in docs]
                        vs.add_documents(docs, ids=ids)
                        total_chunks += len(docs)
                    st.session_state.image_ingest_log.append(
                        {
                            "timestamp": datetime.utcnow().isoformat(timespec="seconds") + "Z",
                            "files": len(uploaded_images),
                            "chunks": total_chunks,
                            "prompt": image_prompt,
                        }
                    )
                    st.success(f"Indexed {len(uploaded_images)} images as {total_chunks} chunks in vector DB.")
                except Exception as exc:
                    st.error(f"Image indexing failed: {exc}")
    with c_ing2:
        if st.session_state.image_ingest_log:
            last = st.session_state.image_ingest_log[-1]
            st.metric("Last Indexed Chunks", last["chunks"])
            st.metric("Last Indexed Files", last["files"])

    st.markdown("---")
    chat_q = st.text_input("Ask chatbot (image-aware)", value="Find products and moodboard matching my uploaded images")
    if st.button("Run Image-Aware Agent Chat", use_container_width=True):
        if not api_key:
            st.error("Set OPENAI_API_KEY first.")
        else:
            try:
                vs = get_image_vector_store(api_key)
                matches = retrieve_image_matches(vs, f"{chat_q}. user_prompt={image_prompt}", top_k=8)
                if not matches:
                    st.warning("No image matches found. Upload and index images first.")
                else:
                    best_image = matches[0]["image_name"]
                    style_seed = f"{image_prompt}. matched_image={best_image}. question={chat_q}"
                    style_seed = f"design_type={design_type}. {style_seed}"
                    style_dna = analyze_style_dna(style_seed)
                    decor = style_dna.get("decor_types", ["rug", "lamp", "console"])
                    products = search_products(style_dna, decor, budget_min, budget_max)
                    board = compose_moodboard(style_dna, products)

                    agent_messages = [
                        {"agent": "Style Agent", "message": f"Matched image: {best_image}; inferred tags: {style_dna.get('style_tags', [])}"},
                        {"agent": "Retail Agent", "message": f"Sourced {len(products)} products within budget {budget_min}-{budget_max}."},
                        {"agent": "Moodboard Agent", "message": f"Composed board with {len(board.get('board_items', []))} items."},
                    ]

                    llm = ChatOpenAI(model=llm_model, temperature=creativity_level, api_key=api_key)
                    final_prompt = (
                        "You are a design orchestrator. Use agent conversation and retrieved image match scores "
                        "to answer the user question and provide final recommendation.\\n\\n"
                        f"User question: {chat_q}\\n"
                        f"Image matches: {matches[:5]}\\n"
                        f"Agent messages: {agent_messages}\\n"
                        f"Style DNA: {style_dna}\\n"
                        f"Top Products: {products[:5]}\\n"
                        f"Moodboard: {board}"
                    )
                    final_answer = llm.invoke(final_prompt).content
                    if enable_live_web:
                        web_result = run_live_web_search(
                            api_key,
                            f"{chat_q}. style={style_dna.get('style_tags', [])}, decor={decor}, budget={budget_min}-{budget_max}",
                            llm_model=llm_model,
                        )
                        final_answer = f"{final_answer}\n\nLive Web Search:\n{web_result}"
                    st.session_state.last_image_answer = final_answer
                    st.session_state.last_image_question = chat_q

                    st.markdown("**Best-Matching Uploaded Images**")
                    st.dataframe(pd.DataFrame(matches), use_container_width=True)

                    st.markdown("**Agent-to-Agent Conversation**")
                    st.dataframe(pd.DataFrame(agent_messages), use_container_width=True)

                    st.markdown("**Final Assistant Recommendation**")
                    st.write(final_answer)
                    render_feedback_controls(
                        scope="image_chat_tab",
                        question=chat_q,
                        answer=final_answer,
                    )

                    st.markdown("**Generated Moodboard Output**")
                    render_board(board)

                    st.session_state.agent_chat_log.append(
                        {
                            "timestamp": datetime.utcnow().isoformat(timespec="seconds") + "Z",
                            "question": chat_q,
                            "best_image": best_image,
                            "final_answer": final_answer,
                            "match_count": len(matches),
                        }
                    )
            except Exception as exc:
                st.error(f"Image-aware chat failed: {exc}")

    if st.session_state.agent_chat_log:
        st.markdown("**Chat History (Image-Aware)**")
        st.dataframe(pd.DataFrame(st.session_state.agent_chat_log), use_container_width=True)

    if st.session_state.feedback_log:
        st.markdown("**Feedback Log**")
        st.dataframe(pd.DataFrame(st.session_state.feedback_log), use_container_width=True)
