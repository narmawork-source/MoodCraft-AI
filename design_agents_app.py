import os
import time
import uuid
import base64
import json
import requests
from collections import defaultdict
from datetime import datetime
from io import BytesIO
from typing import Any

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
    radial-gradient(1200px 500px at 95% -10%, #eaf4ff 0%, rgba(234,244,255,0) 62%),
    linear-gradient(180deg, #fcfdff 0%, #f7fbff 100%);
  color: var(--ink) !important;
}
.stApp [data-testid="stAppViewContainer"],
.stApp [data-testid="stMainBlockContainer"],
.stApp [data-testid="stSidebar"],
.stApp [data-testid="stMarkdownContainer"],
.stApp [data-testid="stText"],
.stApp [data-testid="stMetricLabel"],
.stApp [data-testid="stMetricValue"],
.stApp p,
.stApp span,
.stApp div,
.stApp li,
.stApp label,
.stApp h1,
.stApp h2,
.stApp h3,
.stApp h4,
.stApp h5,
.stApp h6 {
  color: var(--ink) !important;
}
.stApp [data-testid="stSidebar"] * {
  color: var(--ink) !important;
}
.stApp [data-testid="stSidebar"],
.stApp [data-testid="stSidebarContent"] {
  background: #ffffff !important;
  border-right: 1px solid #b7cde8;
}
.block-container { padding-top: 1.2rem; max-width: 1200px; }
.mc-hero {
  background:
    linear-gradient(135deg, rgba(235,245,255,.96) 0%, rgba(221,236,252,.96) 48%, rgba(207,226,246,.96) 100%),
    url('https://images.unsplash.com/photo-1616594039964-3dd9f9f142c0?auto=format&fit=crop&w=1200&q=60');
  background-size: cover;
  border-radius: 24px;
  padding: 1.6rem 1.5rem;
  margin-bottom: 1rem;
  box-shadow: 0 12px 30px rgba(21,31,58,.18);
  border: 1px solid #b7cde8;
}
.mc-title { color: #1f2a3d; font-size: 2.2rem; font-weight: 700; margin: 0; letter-spacing: .2px; }
.mc-sub { color: #4b5a69; margin-top: .35rem; max-width: 700px; }
.mc-kpis {
  display: grid;
  grid-template-columns: repeat(4, minmax(0, 1fr));
  gap: .55rem;
  margin-top: 1rem;
}
.mc-kpi {
  background: rgba(245, 251, 255, .92);
  border: 1px solid #bcd0ea;
  border-radius: 12px;
  color: #1f2a3d;
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
  border: 1px solid #c2d7ef;
  border-radius: 14px;
  padding: .85rem .95rem;
  box-shadow: 0 6px 14px rgba(31, 56, 91, .06);
}
.mc-step-card {
  background: #ffffff;
  border: 1px solid #b7cde8;
  border-radius: 14px;
  padding: .8rem .9rem;
  box-shadow: 0 4px 10px rgba(31, 56, 91, .05);
  margin-bottom: .65rem;
}
.mc-step-card.step2 {
  background: #ffffff;
  border-color: #b7cde8;
}
.mc-step-card.step3 {
  background: #ffffff;
  border-color: #b7cde8;
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
  border: 1px solid #b7cde8;
  object-fit: cover;
}
.mc-card {
  background: #ffffff;
  border: 1px solid #b7cde8;
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
  background: #f2f8ff;
  border-left: 4px solid #b7cde8;
  border-radius: 8px;
  padding: .55rem .75rem;
  margin-bottom: .5rem;
}
/* Strong, visible backgrounds for step expanders */
div[data-testid="stExpander"] {
  background: #ffffff;
  border: 1px solid #b7cde8;
  border-radius: 14px;
  padding: 4px 6px;
  margin-bottom: 10px;
}
div[data-testid="stExpander"] details > summary {
  background: linear-gradient(180deg, #f2f8ff 0%, #e8f2ff 100%);
  border: 1px solid #b7cde8;
  border-radius: 10px;
  padding: .4rem .65rem;
}
div[data-testid="stExpander"] details > div {
  background: #ffffff;
  border-radius: 10px;
  padding: .35rem .35rem .2rem .35rem;
}
.mc-arch {
  background: #ffffff;
  border: 1px solid #b7cde8;
  border-radius: 14px;
  padding: .9rem 1rem;
  box-shadow: 0 4px 12px rgba(40,48,67,.05);
}
.mc-arch b { color: #1f2a3d; }
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
  <p class="mc-title">MoodCraft AI</p>
  <p class="mc-sub">Multi-agent interior design copilot that analyzes your room, sources products, composes design options, and measures quality with QAEval.</p>
  <div class="mc-kpis">
    <div class="mc-kpi"><div class="v">4 Agents</div><div class="l">Vision, Style, Retail, Finalizer</div></div>
    <div class="mc-kpi"><div class="v">Image KB</div><div class="l">Chunked + Vector Search</div></div>
    <div class="mc-kpi"><div class="v">QAEval</div><div class="l">Prompt-Level Evaluation</div></div>
    <div class="mc-kpi"><div class="v">Feedback</div><div class="l">Thumbs Up / Down Metrics</div></div>
  </div>
</div>
<div class="mc-front-grid">
  <div class="mc-front-card">
    <h4>1) Upload + Understand</h4>
    <p>Upload your room image and prompt. Vision Agent reads room cues and space constraints.</p>
  </div>
  <div class="mc-front-card">
    <h4>2) Plan + Source</h4>
    <p>Style Agent sets design direction. Retail Curator finds budget-fit accessories with links.</p>
  </div>
  <div class="mc-front-card">
    <h4>3) Generate + Choose</h4>
    <p>Finalizer creates 3 room options. You select favorites and review quality in Diagnostics.</p>
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
if "quick_generated_images" not in st.session_state:
    st.session_state.quick_generated_images = []
if "quick_context" not in st.session_state:
    st.session_state.quick_context = {}
if "usage_log" not in st.session_state:
    st.session_state.usage_log = []
if "creativity_enabled" not in st.session_state:
    st.session_state.creativity_enabled = False
if "creativity_level" not in st.session_state:
    st.session_state.creativity_level = 0.0
if "quick_creativity_level" not in st.session_state:
    st.session_state.quick_creativity_level = 0.0

IMAGE_VDB_DIR = "moodcraft_image_vdb"
IMAGE_COLLECTION = "moodcraft_image_chunks"


@st.cache_data(show_spinner=False)
def load_avatar_data_uri(path: str) -> str:
    with open(path, "rb") as f:
        b = base64.b64encode(f.read()).decode("utf-8")
    return f"data:image/svg+xml;base64,{b}"


def resolve_secret(name: str) -> str:
    try:
        from_secrets = str(st.secrets.get(name, ""))
    except Exception:
        from_secrets = ""
    from_env = os.getenv(name, "")
    return (from_secrets or from_env).strip()


def set_quick_prompt(text: str):
    st.session_state.quick_prompt_text = text


def sync_creativity_from_quick():
    q_val = float(st.session_state.get("quick_creativity_level", 0.0))
    if q_val > 0:
        st.session_state.creativity_enabled = True
        st.session_state.creativity_level = q_val


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


def normalize_reference_image(image_bytes: bytes, max_side: int = 1536) -> bytes:
    """Normalize uploads (jpg/png/etc.) to a consistent PNG for image-edit APIs."""
    img = Image.open(BytesIO(image_bytes)).convert("RGB")
    img.thumbnail((max_side, max_side))
    out = BytesIO()
    img.save(out, format="PNG")
    return out.getvalue()


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


def retrieve_image_cues(vs: Chroma, query: str, top_k: int = 6) -> str:
    docs = vs.similarity_search(query, k=top_k)
    snippets = []
    for d in docs:
        txt = (d.page_content or "").strip()
        if txt:
            snippets.append(txt[:180])
    if not snippets:
        return "no retrieved visual cues"
    joined = " | ".join(snippets[:4])
    return joined[:700]


def _clip(text: str, limit: int) -> str:
    t = text or ""
    if len(t) <= limit:
        return t
    return t[: limit - 3] + "..."


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
    image_backend: str,
    image_model: str,
    creativity: float,
    user_intent: str = "",
    image_context: str = "",
    hf_token: str = "",
    variation_note: str = "",
    lock_to_image_layout: bool = False,
    reference_image_bytes: bytes | None = None,
    use_direct_reference: bool = False,
) -> bytes:
    design_profile = {
        "Modern": "clean lines, neutral tones, minimal clutter",
        "Industrial": "raw textures, black metal accents, exposed structure cues",
        "Contemporary": "soft geometry, layered textures, current trend styling",
        "Traditional": "classic forms, warm woods, balanced symmetry",
        "Minimalist": "very simple forms, negative space, restrained palette",
        "Scandinavian": "light woods, bright airy look, cozy functional decor",
        "Japandi": "zen minimalism, natural materials, low-profile furniture",
        "Bohemian": "eclectic textiles, earthy colors, artisanal accessories",
        "Mid-Century": "tapered legs, walnut/oak tones, iconic retro shapes",
        "Coastal": "light blues, sandy neutrals, breezy natural textures",
        "Farmhouse": "rustic wood, cozy fabrics, practical layered decor",
        "Luxury": "premium finishes, elegant contrast, refined lighting",
    }.get(design_type, "cohesive interior style")
    top_items = ", ".join([p.get("title", "item") for p in products[:5]])
    palette = ", ".join([p.get("hex", "") for p in style_dna.get("palette", [])[:5]])
    mats = ", ".join(style_dna.get("materials", [])[:5])
    prompt = (
        "Create a high-quality photoreal interior render (not text poster). "
        f"MUST STRICTLY follow the selected design type and do not drift to generic modern unless explicitly asked. "
        f"Design type: {design_type}. "
        f"User request: {user_intent or 'interior redesign concept'}. "
        f"Reference from uploaded image retrieval: {image_context or 'no uploaded reference available'}. "
        f"Variation directive: {variation_note or 'base concept'}. "
        f"Design direction details: {design_profile}. "
        f"Style tags: {', '.join(style_dna.get('style_tags', []))}. "
        f"Palette: {palette}. Materials: {mats}. "
        f"Include these accessories in scene: {top_items}. "
        "Output a realistic room scene with cohesive furniture, decor, and lighting. "
        "Do not add text in image."
    )
    if lock_to_image_layout:
        prompt += (
            " Strongly preserve the uploaded room's layout, camera angle, architectural structure, "
            "window/door positions, and major furniture placement while restyling materials/decor."
        )
    if creativity >= 0.65:
        prompt += " Explore bolder composition, richer contrast, and more expressive styling details."
    elif creativity <= 0.25:
        prompt += " Keep the layout highly practical, minimal, and conservative."
    prompt = _clip(prompt, 3900)
    if image_backend == "Diffusion (HF SDXL)":
        if not hf_token:
            raise ValueError("HF token is required for Diffusion backend.")
        headers = {"Authorization": f"Bearer {hf_token}"}
        payload = {
            "inputs": prompt,
            "parameters": {
                "guidance_scale": 8.0,
                "num_inference_steps": 35,
                "negative_prompt": "blurry, low quality, distorted, text overlay, watermark",
            },
        }
        resp = requests.post(
            "https://api-inference.huggingface.co/models/stabilityai/stable-diffusion-xl-base-1.0",
            headers=headers,
            json=payload,
            timeout=120,
        )
        if resp.status_code != 200:
            raise RuntimeError(f"HF diffusion failed: {resp.status_code} {resp.text[:200]}")
        return resp.content

    client = OpenAI(api_key=api_key)
    if (
        use_direct_reference
        and reference_image_bytes
        and image_backend == "OpenAI"
    ):
        try:
            resp = client.images.edit(
                model=image_model,
                image=("reference.png", reference_image_bytes, "image/png"),
                prompt=prompt,
                size="1024x1024",
            )
            b64 = resp.data[0].b64_json
            return base64.b64decode(b64)
        except Exception:
            pass

    resp = client.images.generate(
        model=image_model,
        prompt=prompt,
        size="1024x1024",
    )
    b64 = resp.data[0].b64_json
    return base64.b64decode(b64)


def generate_image_variants(
    count: int,
    api_key: str,
    design_type: str,
    style_dna: dict,
    products: list[dict],
    image_backend: str,
    image_model: str,
    creativity: float,
    user_intent: str,
    image_context: str,
    hf_token: str,
    lock_to_image_layout: bool = False,
    reference_image_bytes: bytes | None = None,
    use_direct_reference: bool = False,
) -> list[bytes]:
    notes = [
        f"{design_type} variant A: symmetrical composition, calm lighting, restrained accents",
        f"{design_type} variant B: asymmetrical composition, focal accent corner, layered textures",
        f"{design_type} variant C: bolder decor contrast, alternate furniture arrangement",
    ]
    out: list[bytes] = []
    for i in range(count):
        out.append(
            generate_moodboard_image(
                api_key=api_key,
                design_type=design_type,
                style_dna=style_dna,
                products=products,
                image_backend=image_backend,
                image_model=image_model,
                creativity=creativity,
                user_intent=user_intent,
                image_context=image_context,
                hf_token=hf_token,
                variation_note=notes[i % len(notes)],
                lock_to_image_layout=lock_to_image_layout,
                reference_image_bytes=reference_image_bytes,
                use_direct_reference=use_direct_reference,
            )
        )
    return out


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


def parse_json_response(text: str) -> dict[str, Any]:
    t = (text or "").strip()
    if t.startswith("```"):
        t = t.strip("`")
        if t.lower().startswith("json"):
            t = t[4:].strip()
    return json.loads(t)


def llm_json_step(api_key: str, llm_model: str, role: str, schema_hint: str, payload: dict, fallback: dict) -> dict:
    try:
        llm = ChatOpenAI(model=llm_model, temperature=0, api_key=api_key)
        prompt = (
            f"You are {role}. Return strict JSON only. No markdown.\n"
            f"Schema hint: {schema_hint}\n"
            f"Input payload: {json.dumps(payload)}"
        )
        raw = llm.invoke(prompt).content
        return parse_json_response(raw)
    except Exception:
        return fallback


def run_three_agent_consensus(
    api_key: str,
    llm_model: str,
    user_prompt: str,
    design_type: str,
    image_matches: list[dict],
    selected_decor_types: list[str],
    budget_min: int,
    budget_max: int,
    image_cues: str = "",
) -> dict:
    vision_fallback = {
        "style_cues": [design_type.lower(), "natural_textures"],
        "palette_detected": ["#f5f1ea", "#d8c9b1", "#8b6f47"],
        "constraints": ["respect uploaded image composition"],
        "existing_items": ["current furniture"],
    }
    vision = llm_json_step(
        api_key=api_key,
        llm_model=llm_model,
        role="Vision Agent",
        schema_hint='{"style_cues":[],"palette_detected":[],"constraints":[],"existing_items":[]}',
        payload={"user_prompt": user_prompt, "image_matches": image_matches[:5], "design_type": design_type},
        fallback=vision_fallback,
    )

    style_fallback = {
        "target_style": design_type.lower(),
        "must_keep": [],
        "avoid": ["clutter"],
        "do": ["cohesive styling", "balanced layout"],
        "budget": {"min": budget_min, "max": budget_max},
    }
    style = llm_json_step(
        api_key=api_key,
        llm_model=llm_model,
        role="Style Agent",
        schema_hint='{"target_style":"","must_keep":[],"avoid":[],"do":[],"budget":{"min":0,"max":0}}',
        payload={
            "user_prompt": user_prompt,
            "vision": vision,
            "design_type": design_type,
            "budget": [budget_min, budget_max],
            "image_cues": image_cues,
        },
        fallback=style_fallback,
    )

    retail_fallback = {
        "categories": selected_decor_types or ["rug", "lamp", "console"],
        "material_preferences": ["oak", "linen"],
        "size_constraints": {},
        "retailer_queries": [f"{design_type} decor under budget"],
        "alternatives": {},
    }
    retail = llm_json_step(
        api_key=api_key,
        llm_model=llm_model,
        role="Retail Curator Agent",
        schema_hint='{"categories":[],"material_preferences":[],"size_constraints":{},"retailer_queries":[],"alternatives":{}}',
        payload={"vision": vision, "style": style, "user_prompt": user_prompt, "image_cues": image_cues},
        fallback=retail_fallback,
    )

    style_seed = (
        f"{design_type} {user_prompt} {style.get('target_style', design_type)} "
        f"{' '.join(vision.get('style_cues', []))} image_cues={image_cues}"
    )
    style_seed = f"design_type={design_type}. target_style={style.get('target_style', design_type)}. {style_seed}"
    style_dna = analyze_style_dna(style_seed)
    decor = selected_decor_types or retail.get("categories", []) or style_dna.get("decor_types", ["rug", "lamp", "console"])
    products = search_products(style_dna, decor, budget_min, budget_max)
    board = compose_moodboard(style_dna, products)

    synthesis_fallback = {
        "final_design_brief": f"{design_type} direction aligned to uploaded room context and prompt.",
        "resolved_conflicts": ["balanced minimal styling and essential decor only"],
        "shopping_checklist": decor[:6],
        "recommended_direction": "Option with strongest style-context match",
    }
    synthesis = llm_json_step(
        api_key=api_key,
        llm_model=llm_model,
        role="Synthesis Agent",
        schema_hint='{"final_design_brief":"","resolved_conflicts":[],"shopping_checklist":[],"recommended_direction":""}',
        payload={
            "vision": vision,
            "style": style,
            "retail": retail,
            "products": products[:6],
            "board": board,
        },
        fallback=synthesis_fallback,
    )

    messages = [
        {"agent": "Vision Agent", "message": f"Detected cues={vision.get('style_cues', [])}, constraints={vision.get('constraints', [])}"},
        {"agent": "Style Agent", "message": f"Target={style.get('target_style', design_type)}, avoid={style.get('avoid', [])}"},
        {"agent": "Retail Curator Agent", "message": f"Categories={retail.get('categories', decor)}, queries={retail.get('retailer_queries', [])[:2]}"},
        {"agent": "Synthesis Agent", "message": f"Brief={synthesis.get('final_design_brief', '')}"},
    ]
    return {
        "vision": vision,
        "style": style,
        "retail": retail,
        "synthesis": synthesis,
        "style_dna": style_dna,
        "products": products,
        "board": board,
        "decor": decor,
        "agent_messages": messages,
    }


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
    api_key = resolve_secret("OPENAI_API_KEY")
    hf_token = resolve_secret("HF_API_TOKEN")
    if api_key:
        st.caption("OpenAI key loaded from secrets/env.")
    else:
        st.warning("Missing OPENAI_API_KEY in Streamlit secrets or environment.")
    with st.expander("Settings", expanded=True):
        image_backend = st.selectbox("Image Backend", ["OpenAI", "Diffusion (HF SDXL)"], index=0)
        if image_backend == "Diffusion (HF SDXL)" and not hf_token:
            st.caption("Missing HF_API_TOKEN in secrets/env for diffusion backend.")
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
        creativity_enabled = st.toggle("Creativity Mode", key="creativity_enabled")
        creativity_level = st.slider(
            "Creativity Level",
            0.0,
            1.0,
            float(st.session_state.creativity_level),
            0.05,
            disabled=not creativity_enabled,
            key="creativity_level",
        )
        design_type = st.selectbox(
            "Design Type",
            [
                "Modern",
                "Industrial",
                "Contemporary",
                "Traditional",
                "Minimalist",
                "Scandinavian",
                "Japandi",
                "Bohemian",
                "Mid-Century",
                "Coastal",
                "Farmhouse",
                "Luxury",
            ],
            index=0,
        )
        budget_min, budget_max = st.slider("Budget Range", 20, 1500, (100, 400), step=10)
        decor_options = [
            "rug",
            "lamp",
            "console",
            "chair",
            "sofa",
            "coffee table",
            "wall art",
            "curtains",
            "mirror",
            "plants",
            "storage",
        ]
        selected_decor_types = st.multiselect(
            "Decor Types",
            options=decor_options,
            default=["rug", "lamp", "console"],
        )
        fast_mode = st.toggle("Fast Mode (quicker response)", value=False)
        lock_to_image_layout = st.toggle("Lock to Uploaded Image Layout", value=False)
        use_direct_reference = st.toggle("Use Uploaded Image as Direct Reference (OpenAI)", value=True)
        auto_qaeval = st.checkbox("Auto QAEval", value=True)

    st.markdown("---")
    with st.expander("Quick Workflow", expanded=False):
        st.markdown('<div class="mc-step">1) Enter prompt.</div>', unsafe_allow_html=True)
        st.markdown('<div class="mc-step">2) Upload room/inspiration image.</div>', unsafe_allow_html=True)
        st.markdown('<div class="mc-step">3) Click Generate My Design.</div>', unsafe_allow_html=True)
        st.markdown('<div class="mc-step">4) Review images, products, and diagnostics.</div>', unsafe_allow_html=True)


style_avatar_uri = load_avatar_data_uri("design_agents/assets/style_avatar.svg")
retail_avatar_uri = load_avatar_data_uri("design_agents/assets/retail_avatar.svg")
board_avatar_uri = load_avatar_data_uri("design_agents/assets/board_avatar.svg")
orchestrator_avatar_uri = load_avatar_data_uri("design_agents/assets/orchestrator_avatar.svg")

tab_design, tab_diag, tab_agents, tab_arch = st.tabs(["Design Studio", "Diagnostics", "Meet the Agents", "Architecture"])

with tab_design:
    st.subheader("Quick Design Studio")
    st.caption("Follow 3 simple steps: write request, upload images, generate.")
    if fast_mode:
        st.info("Fast Mode is ON: fewer retrieval docs, fewer generated images, and QAEval is skipped for speed.")

    if "quick_prompt_text" not in st.session_state:
        st.session_state.quick_prompt_text = ""

    with st.expander("Step 1: Describe Your Design Goal", expanded=True):
        st.markdown('<div class="mc-step-card">', unsafe_allow_html=True)
        st.markdown("**Step 1: Type your design request below**")
        q_prompt = st.text_area(
            "Describe what you want to change",
            key="quick_prompt_text",
            height=120,
            help="Type your request or choose an example and edit it.",
            placeholder="Example: Redesign my room as a modern cozy space with warm lights, practical storage, and a neutral color palette.",
        )
        st.markdown("</div>", unsafe_allow_html=True)
        st.caption("Try an example, then edit it:")
        ex_cols = st.columns(3)
        with ex_cols[0]:
            st.button(
                "Modern + Cozy",
                use_container_width=True,
                key="prompt_ex_modern",
                on_click=set_quick_prompt,
                args=("Design a modern cozy living room with warm lights, neutral rug, and wood accents.",),
            )
        with ex_cols[1]:
            st.button(
                "Pooja Room",
                use_container_width=True,
                key="prompt_ex_pooja",
                on_click=set_quick_prompt,
                args=("Create a serene pooja room with traditional elements, soft lighting, and compact storage.",),
            )
        with ex_cols[2]:
            st.button(
                "Luxury Refresh",
                use_container_width=True,
                key="prompt_ex_luxury",
                on_click=set_quick_prompt,
                args=("Give this room a luxury refresh with elegant textures, statement lighting, and premium finishes.",),
            )
        st.caption("You can also ask edits like: 'Make it brighter', 'Add more storage', 'Keep same layout but change to japandi'.")

    with st.expander("Step 2: Upload Room / Inspiration Images", expanded=True):
        st.markdown('<div class="mc-step-card step2">', unsafe_allow_html=True)
        st.markdown("**Step 2: Upload room and/or inspiration images**")
        upload_mode = st.radio(
            "Uploaded Image Intent",
            ["My Room (visualize my room)", "Inspiration Sample (style idea)", "Both"],
            horizontal=True,
        )
        room_upload = st.file_uploader(
            "Upload your room image",
            type=["jpg", "jpeg", "png"],
            accept_multiple_files=False,
            key="quick_room_uploader",
        )
        inspiration_uploads = st.file_uploader(
            "Upload inspiration image(s)",
            type=["jpg", "jpeg", "png"],
            accept_multiple_files=True,
            key="quick_inspo_uploader",
        )
        retain_structure = st.toggle(
            "Retain windows/doors/layout from my room image",
            value=True if upload_mode in {"My Room (visualize my room)", "Both"} else False,
        )
        if use_direct_reference and not room_upload and not inspiration_uploads:
            st.caption("Direct reference mode is ON. Upload at least one image to apply reference-guided generation.")
        st.markdown("</div>", unsafe_allow_html=True)

    with st.expander("Step 3: Generation Options", expanded=False):
        st.markdown('<div class="mc-step-card step3">', unsafe_allow_html=True)
        st.markdown("**Step 3: Tune generation speed and creativity**")
        q_creativity = st.slider(
            "Creativity (Quick Flow)",
            0.0,
            1.0,
            float(st.session_state.quick_creativity_level),
            0.05,
            key="quick_creativity_level",
            on_change=sync_creativity_from_quick,
        )
        st.caption("Lower creativity = practical designs. Higher creativity = bolder ideas.")
        st.markdown("</div>", unsafe_allow_html=True)

    if st.button("Generate My Design", use_container_width=True):
        if not api_key:
            st.error("Set OPENAI_API_KEY first.")
        elif not q_prompt.strip():
            st.error("Enter a design prompt.")
        else:
            try:
                matches = []
                image_cues = "no uploaded visual cues"
                reference_image_bytes = None
                failed_uploads = []
                uploads_for_index = []
                if room_upload is not None:
                    uploads_for_index.append(room_upload)
                if inspiration_uploads:
                    uploads_for_index.extend(inspiration_uploads)

                if uploads_for_index:
                    if upload_mode in {"My Room (visualize my room)", "Both"} and room_upload is not None:
                        try:
                            reference_image_bytes = normalize_reference_image(room_upload.getvalue())
                        except Exception:
                            reference_image_bytes = room_upload.getvalue()
                    elif upload_mode == "Inspiration Sample (style idea)" and inspiration_uploads:
                        try:
                            reference_image_bytes = normalize_reference_image(inspiration_uploads[0].getvalue())
                        except Exception:
                            reference_image_bytes = inspiration_uploads[0].getvalue()

                    vs = get_image_vector_store(api_key)
                    indexed_any = False
                    for f in uploads_for_index:
                        try:
                            docs = chunk_image_to_documents(f.getvalue(), f.name, q_prompt, grid=3)
                            if not docs:
                                failed_uploads.append(f"{f.name} (no readable content)")
                                continue
                            ids = [str(uuid.uuid4()) for _ in docs]
                            vs.add_documents(docs, ids=ids)
                            indexed_any = True
                        except Exception as file_exc:
                            failed_uploads.append(f"{f.name} ({str(file_exc)[:80]})")
                            continue
                    if indexed_any:
                        retrieval_k = 4 if fast_mode else 8
                        cue_k = 3 if fast_mode else 6
                        matches = retrieve_image_matches(vs, q_prompt, top_k=retrieval_k)
                        image_cues = retrieve_image_cues(vs, q_prompt, top_k=cue_k)

                matched_image = matches[0]["image_name"] if matches else "none"
                effective_lock_to_layout = (
                    retain_structure
                    and room_upload is not None
                    and upload_mode in {"My Room (visualize my room)", "Both"}
                )
                effective_direct_reference = bool(use_direct_reference and reference_image_bytes is not None)
                pipeline = run_three_agent_consensus(
                    api_key=api_key,
                    llm_model=llm_model,
                    user_prompt=q_prompt,
                    design_type=design_type,
                    image_matches=matches,
                    selected_decor_types=selected_decor_types,
                    budget_min=budget_min,
                    budget_max=budget_max,
                    image_cues=image_cues,
                )
                style_dna = pipeline["style_dna"]
                products = pipeline["products"]
                board = pipeline["board"]
                decor = pipeline["decor"]
                synthesis = pipeline["synthesis"]
                agent_messages = pipeline["agent_messages"]
                st.session_state.style_dna = style_dna
                st.session_state.products = products
                st.session_state.board = board
                final_design_brief = (
                    f"{synthesis.get('final_design_brief', '')}. "
                    f"Recommended direction: {synthesis.get('recommended_direction', '')}. "
                    f"Checklist: {synthesis.get('shopping_checklist', [])}."
                ).strip()
                generation_intent = _clip(f"{q_prompt}. {final_design_brief}", 900)

                with st.spinner("Generating 3 design options..."):
                    variant_count = 1 if fast_mode else 3
                    quick_images = generate_image_variants(
                        count=variant_count,
                        api_key=api_key,
                        design_type=design_type,
                        style_dna=style_dna,
                        products=products,
                        image_backend=image_backend,
                        image_model=image_model,
                        creativity=q_creativity,
                        user_intent=generation_intent,
                        image_context=(
                            f"upload_mode={upload_mode}; retain_structure={effective_lock_to_layout}; "
                            f"best_match={matched_image}; scores={str(matches[:3])[:300]}; "
                            f"retrieved_visual_cues={image_cues}"
                        )
                        if matches
                        else _clip(
                            f"upload_mode={upload_mode}; retain_structure={effective_lock_to_layout}; "
                            f"no match; retrieved_visual_cues={image_cues}",
                            700,
                        ),
                        hf_token=hf_token,
                        lock_to_image_layout=effective_lock_to_layout or lock_to_image_layout,
                        reference_image_bytes=reference_image_bytes,
                        use_direct_reference=effective_direct_reference,
                    )

                st.session_state.quick_generated_images = quick_images
                st.session_state.quick_context = {
                    "prompt": q_prompt,
                    "matches": matches,
                    "failed_uploads": failed_uploads,
                    "matched_image": matched_image,
                    "image_cues": image_cues,
                    "style_dna": style_dna,
                    "products": products,
                    "board": board,
                    "agent_messages": agent_messages,
                    "synthesis": synthesis,
                    "final_design_brief": final_design_brief,
                    "generation_intent": generation_intent,
                    "upload_mode": upload_mode,
                    "retain_structure": effective_lock_to_layout,
                }
                st.session_state.usage_log.append(
                    {
                        "timestamp": datetime.utcnow().isoformat(timespec="seconds") + "Z",
                        "prompt": q_prompt,
                        "design_type": design_type,
                        "decor_types": ", ".join(decor),
                        "product_count": len(products),
                        "top_product": products[0]["title"] if products else "",
                        "conflict_count": len(synthesis.get("resolved_conflicts", [])),
                    }
                )
                if auto_qaeval and api_key and not fast_mode:
                    prediction = (
                        f"Final brief: {synthesis.get('final_design_brief', '')}. "
                        f"Resolved conflicts: {synthesis.get('resolved_conflicts', [])}. "
                        f"Top products: {[p.get('title') for p in products[:3]]}."
                    )
                    reference = (
                        f"Response should satisfy prompt '{q_prompt}', align with design type '{design_type}', "
                        f"respect budget range {budget_min}-{budget_max}, and include practical decor guidance."
                    )
                    ev = eval_answer(api_key, q_prompt, reference, prediction)
                    agent_eval = evaluate_agents(
                        api_key=api_key,
                        llm_model=llm_model,
                        question=q_prompt,
                        style_dna=style_dna,
                        products=products,
                        board=board,
                        final_answer=prediction,
                    )
                    st.session_state.qa_log.append(
                        {
                            "question": q_prompt,
                            "reference": reference,
                            "prediction": prediction,
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
                            "question": q_prompt,
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

            except Exception as exc:
                st.error(f"Quick design generation failed: {exc}")

    if st.session_state.quick_generated_images:
        qctx = st.session_state.quick_context
        option_count = len(st.session_state.quick_generated_images)
        st.markdown(f"### Your New Room Concepts ({option_count} Option{'s' if option_count != 1 else ''})")
        img_cols = st.columns(option_count)
        for i, img_bytes in enumerate(st.session_state.quick_generated_images):
            with img_cols[i]:
                st.image(img_bytes, caption=f"Option {i + 1}: {qctx.get('prompt', '')}", use_container_width=True)
                st.download_button(
                    f"Download Option {i + 1}",
                    data=img_bytes,
                    file_name=f"quick_design_option_{i + 1}.png",
                    mime="image/png",
                    use_container_width=True,
                    key=f"quick_download_{i}",
                )
                c_up, c_down = st.columns(2)
                with c_up:
                    if st.button("👍", key=f"quick_img_up_{i}", use_container_width=True):
                        st.session_state.feedback_log.append(
                            {
                                "timestamp": datetime.utcnow().isoformat(timespec="seconds") + "Z",
                                "scope": "quick_image_gallery",
                                "question": qctx.get("prompt", ""),
                                "answer": f"option_{i+1}",
                                "feedback": "up",
                            }
                        )
                with c_down:
                    if st.button("👎", key=f"quick_img_down_{i}", use_container_width=True):
                        st.session_state.feedback_log.append(
                            {
                                "timestamp": datetime.utcnow().isoformat(timespec="seconds") + "Z",
                                "scope": "quick_image_gallery",
                                "question": qctx.get("prompt", ""),
                                "answer": f"option_{i+1}",
                                "feedback": "down",
                            }
                        )

        favorite_opts = [f"Option {i+1}" for i in range(option_count)]
        favorite = st.radio("Pick your favorite image option", favorite_opts, horizontal=True)
        if st.button("Save Favorite", use_container_width=False, key="quick_favorite_btn"):
            st.session_state.feedback_log.append(
                {
                    "timestamp": datetime.utcnow().isoformat(timespec="seconds") + "Z",
                    "scope": "quick_image_gallery",
                    "question": qctx.get("prompt", ""),
                    "answer": favorite.lower().replace(" ", "_"),
                    "feedback": "favorite",
                }
            )
            st.success(f"Saved {favorite} as favorite.")

        matches = qctx.get("matches", [])
        failed_uploads = qctx.get("failed_uploads", [])
        if failed_uploads:
            st.warning("Some uploaded files were skipped:\n- " + "\n- ".join(failed_uploads))
        if matches:
            st.markdown("### Closest Match From Your Uploaded Images")
            st.dataframe(pd.DataFrame(matches), use_container_width=True)
            if qctx.get("image_cues"):
                st.markdown("### Retrieved Visual Cues Used")
                st.caption(qctx.get("image_cues"))

        products = qctx.get("products", [])
        board = qctx.get("board", {})
        style_dna = qctx.get("style_dna", {})
        st.markdown("### Accessories Needed (With Shopping Links & Pricing)")
        if products:
            prod_df = pd.DataFrame(products)[["title", "retailer", "price", "match_score", "url"]]
            st.dataframe(prod_df, use_container_width=True)
            for p in products[:6]:
                st.markdown(
                    f"- **{p['title']}** | {p['retailer']} | `${p['price']}` | [Buy Link]({p['url']})"
                )
        else:
            st.warning("No products found in selected budget range. Increase the budget range in sidebar.")

        st.markdown("### Agent Summary")
        st.write(
            f"Style Agent identified: {', '.join(style_dna.get('style_tags', []))}. "
            f"Retail Agent sourced {len(products)} items. "
            f"Moodboard Agent prepared {len(board.get('board_items', []))} placements."
        )
        if qctx.get("generation_intent"):
            st.markdown("### Generation Prompt Used (Aligned Output)")
            st.caption(qctx.get("generation_intent"))
        if qctx.get("upload_mode"):
            st.caption(
                f"Generation mode: {qctx.get('upload_mode')} | "
                f"Retain structure: {qctx.get('retain_structure')}"
            )
        if qctx.get("synthesis"):
            st.markdown("### Final Design Brief (Consensus)")
            st.write(qctx["synthesis"].get("final_design_brief", ""))
            if qctx["synthesis"].get("resolved_conflicts"):
                st.write(f"Resolved conflicts: {qctx['synthesis'].get('resolved_conflicts', [])}")
        if qctx.get("agent_messages"):
            st.markdown("### Agent-to-Agent Conversation")
            st.dataframe(pd.DataFrame(qctx["agent_messages"]), use_container_width=True)

with tab_agents:
    st.subheader("Meet the Agents")
    st.caption("Simple explanation of who does what and how they work together to design your room.")

    c1, c2 = st.columns(2)
    with c1:
        st.markdown('<div class="mc-card">', unsafe_allow_html=True)
        st.markdown(
            f"""
            <div class="mc-agent-head">
              <img class="mc-avatar" src="{style_avatar_uri}" alt="Vision Agent Avatar" />
              <h3 style="margin:0">1) Vision Agent</h3>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.write("Looks at your uploaded room image and understands what is already in the room.")
        st.write("Finds visual cues like style, colors, and space limitations.")
        st.markdown("</div>", unsafe_allow_html=True)

    with c2:
        st.markdown('<div class="mc-card">', unsafe_allow_html=True)
        st.markdown(
            f"""
            <div class="mc-agent-head">
              <img class="mc-avatar" src="{retail_avatar_uri}" alt="Style Agent Avatar" />
              <h3 style="margin:0">2) Style Agent</h3>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.write("Understands your request (for example: 'modern pooja room').")
        st.write("Creates a clear style plan including what to include and avoid.")
        st.markdown("</div>", unsafe_allow_html=True)

    c3, c4 = st.columns(2)
    with c3:
        st.markdown('<div class="mc-card">', unsafe_allow_html=True)
        st.markdown(
            f"""
            <div class="mc-agent-head">
              <img class="mc-avatar" src="{board_avatar_uri}" alt="Retail Agent Avatar" />
              <h3 style="margin:0">3) Retail Curator Agent</h3>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.write("Builds the shopping list based on your style and budget.")
        st.write("Returns accessories with price and links that match your room direction.")
        st.markdown("</div>", unsafe_allow_html=True)

    with c4:
        st.markdown('<div class="mc-card">', unsafe_allow_html=True)
        st.markdown(
            f"""
            <div class="mc-agent-head">
              <img class="mc-avatar" src="{orchestrator_avatar_uri}" alt="Finalizer Agent Avatar" />
              <h3 style="margin:0">4) Finalizer Agent</h3>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.write("Combines all agent inputs and decides the final design direction.")
        st.write("Generates 3 image options and final recommendations.")
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("### How They Talk to Each Other")
    st.markdown(
        """
        <div class="mc-arch">
          You upload image + prompt -> Vision Agent reads the room -> Style Agent defines the direction -> 
          Retail Agent picks matching items in budget -> Finalizer Agent combines everything and creates final outputs.
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("### How Budget and Creativity Affect Results")
    st.write("- Budget controls which products are selected.")
    st.write("- Creativity controls how bold or practical the generated images look.")

with tab_arch:
    st.subheader("Architecture")
    st.caption("Simple view of how your request becomes final room designs and shopping suggestions.")
    st.markdown("### Easy Flow")
    st.markdown(
        """
        <div class="mc-arch">
          <b>1) You Describe Goal</b>: Prompt + room/inspiration images<br/>
          <b>2) System Understands Context</b>: Reads visual cues from uploaded images<br/>
          <b>3) Agents Plan Design</b>: Vision -> Style -> Retail -> Finalizer<br/>
          <b>4) You Get Results</b>: 3 design images + accessories with links<br/>
          <b>5) You Improve</b>: Choose favorite and iterate with new prompt edits
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown("### What Each Part Does")
    st.write("- Prompt + Upload: defines user intent and room context.")
    st.write("- Agent Collaboration: creates a design plan and product shortlist.")
    st.write("- Image Generation: shows room options to help visualization.")
    st.write("- Diagnostics: tracks quality and user feedback automatically.")

    with st.expander("Technical Details (Optional)"):
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("### Models Used")
            st.write(f"- LLM (configurable): `{llm_model}`")
            st.write(f"- Image backend: `{image_backend}`")
            st.write(f"- Image model (OpenAI mode): `{image_model}`")
            st.write("- Embeddings: `text-embedding-3-small`")
            st.write("- QAEval judge: `gpt-4.1-mini`")
        with c2:
            st.markdown("### MCP + Storage")
            st.write("- MCP servers provide style/retail/board tool interfaces.")
            st.write("- Uploaded images are chunked and indexed in vector DB.")
            st.write("- Retrieved cues are used during agent planning and generation.")

        st.markdown("### Key Files")
        st.code(
            "design_agents_app.py\n"
            "design_agents/style_agent.py\n"
            "design_agents/retail_agent.py\n"
            "design_agents/moodboard_agent.py\n"
            "mcp_style_server.py\n"
            "mcp_retail_server.py\n"
            "mcp_board_server.py",
            language="text",
        )

if False:
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

if False:
    st.subheader("Retail Sourcing")
    if st.button("Run Retail Sourcing", use_container_width=True):
        dna = st.session_state.style_dna
        if not dna:
            st.error("Run Agent 1 first.")
        else:
            decor = selected_decor_types or ["rug", "lamp", "console"]
            started = time.perf_counter()
            products = search_products(dna, decor, budget_min, budget_max)
            elapsed = int((time.perf_counter() - started) * 1000)
            trace("search_products", True, elapsed)
            st.session_state.products = products

    if st.session_state.products:
        render_products(st.session_state.products)

if False:
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
                    image_backend=image_backend,
                    image_model=image_model,
                    creativity=creativity_level,
                    user_intent=f"{design_type} interior moodboard",
                    image_context="manual moodboard generation tab",
                    hf_token=hf_token,
                    lock_to_image_layout=lock_to_image_layout,
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

with tab_diag:
    st.subheader("Diagnostics (Automatic QAEval)")
    if st.session_state.usage_log:
        udf = pd.DataFrame(st.session_state.usage_log)
        up_count = len(udf)
        avg_products = float(udf["product_count"].mean()) if "product_count" in udf.columns else 0.0
        top_design = udf["design_type"].value_counts().idxmax() if "design_type" in udf.columns else "N/A"
        top_product = udf["top_product"].value_counts().idxmax() if "top_product" in udf.columns and len(udf["top_product"].dropna()) else "N/A"
        avg_conflicts = float(udf["conflict_count"].mean()) if "conflict_count" in udf.columns else 0.0
        k1, k2, k3, k4, k5 = st.columns(5)
        k1.metric("Design Requests", up_count)
        k2.metric("Avg Products/Request", f"{avg_products:.1f}")
        k3.metric("Top Design Type", top_design)
        k4.metric("Most Suggested Product", top_product)
        k5.metric("Avg Resolved Conflicts", f"{avg_conflicts:.1f}")
    st.caption("QAEval runs automatically for every `Generate My Design` request. No manual ask required.")

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**QAEval (Clear Evaluation View)**")
        if st.session_state.qa_log:
            df = pd.DataFrame(st.session_state.qa_log)
            total = len(df)
            correct = int((df["qaeval_grade"] == "CORRECT").sum())
            incorrect = total - correct
            pass_rate = (correct / total * 100) if total else 0.0
            m1, m2, m3 = st.columns(3)
            m1.metric("Pass Rate", f"{pass_rate:.2f}%")
            m2.metric("Passed", correct)
            m3.metric("Failed", incorrect)

            view_df = df.copy()
            view_df["evaluation_result"] = view_df["qaeval_grade"]
            view_df["expected_answer"] = view_df["reference"].astype(str).str.slice(0, 240)
            view_df["model_answer"] = view_df["prediction"].astype(str).str.slice(0, 240)
            view_df["why"] = view_df["qaeval_reason"].astype(str).str.slice(0, 240)

            def _confidence(reason: str) -> str:
                r = (reason or "").lower()
                if any(x in r for x in ["uncertain", "partially", "missing", "not enough"]):
                    return "Low"
                if any(x in r for x in ["clearly", "strong", "well aligned", "matches"]):
                    return "High"
                return "Medium"

            view_df["confidence"] = view_df["why"].apply(_confidence)
            show_failed_only = st.checkbox("Show failed evaluations only", value=False)
            if show_failed_only:
                view_df = view_df[view_df["evaluation_result"] != "CORRECT"]

            st.caption("Each row shows: user question, expected answer, model answer, evaluation result, confidence, and why.")
            st.dataframe(
                view_df[
                    [
                        "question",
                        "expected_answer",
                        "model_answer",
                        "evaluation_result",
                        "confidence",
                        "why",
                    ]
                ],
                use_container_width=True,
            )
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

if False:
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
                    decor = selected_decor_types or style_dna.get("decor_types", ["rug", "lamp", "console"])
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
