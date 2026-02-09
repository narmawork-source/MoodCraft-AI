import os
import time
from datetime import datetime

import pandas as pd
import streamlit as st
from langchain_classic.evaluation.qa import QAEvalChain
from langchain_openai import ChatOpenAI

from design_agents.moodboard_agent import compose_moodboard
from design_agents.retail_agent import search_products
from design_agents.style_agent import analyze_style_dna

st.set_page_config(page_title='MoodCraft AI', page_icon='🛋️', layout='wide')
st.title('MoodCraft AI - 3 Agent Interior Design')

if 'style_dna' not in st.session_state:
    st.session_state.style_dna = {}
if 'products' not in st.session_state:
    st.session_state.products = []
if 'board' not in st.session_state:
    st.session_state.board = {}
if 'qa_log' not in st.session_state:
    st.session_state.qa_log = []
if 'mcp_trace' not in st.session_state:
    st.session_state.mcp_trace = []


def trace(tool: str, ok: bool, ms: int, err: str = ''):
    st.session_state.mcp_trace.append({
        'timestamp': datetime.utcnow().isoformat(timespec='seconds') + 'Z',
        'tool': tool,
        'status': 'success' if ok else 'error',
        'latency_ms': ms,
        'error': err,
    })


def eval_answer(api_key: str, question: str, reference: str, prediction: str) -> dict:
    llm = ChatOpenAI(model='gpt-4.1-mini', temperature=0, api_key=api_key)
    chain = QAEvalChain.from_llm(llm)
    out = chain.evaluate(
        examples=[{'query': question, 'answer': reference}],
        predictions=[{'result': prediction}],
        question_key='query',
        answer_key='answer',
        prediction_key='result',
    )
    txt = (out[0].get('text') or str(out[0])).strip()
    up = txt.upper()
    g = 'INCORRECT'
    if 'CORRECT' in up and 'INCORRECT' not in up:
        g = 'CORRECT'
    return {'grade': g, 'reason': txt}


with st.sidebar:
    api_key = st.text_input('OpenAI API Key', value=os.getenv('OPENAI_API_KEY', ''), type='password')
    budget_min, budget_max = st.slider('Budget', 20, 1500, (100, 400), step=10)
    decor_text = st.text_input('Decor types (comma)', value='rug,lamp,console')
    auto_qaeval = st.checkbox('Auto QAEval', value=True)


t1, t2, t3, t4 = st.tabs(['Agent 1 Style', 'Agent 2 Retail', 'Agent 3 Moodboard', 'QAEval + Trace'])

with t1:
    prompt = st.text_input('Style prompt', value='cozy modern japandi')
    if st.button('Run Style Interpreter'):
        s = time.perf_counter()
        dna = analyze_style_dna(prompt)
        ms = int((time.perf_counter() - s) * 1000)
        trace('analyze_room_images', True, ms)
        st.session_state.style_dna = dna
    if st.session_state.style_dna:
        st.json(st.session_state.style_dna)

with t2:
    if st.button('Run Retail Sourcing'):
        dna = st.session_state.style_dna
        if not dna:
            st.error('Run Agent 1 first')
        else:
            decor = [x.strip() for x in decor_text.split(',') if x.strip()]
            s = time.perf_counter()
            products = search_products(dna, decor, budget_min, budget_max)
            ms = int((time.perf_counter() - s) * 1000)
            trace('search_products', True, ms)
            st.session_state.products = products
    if st.session_state.products:
        st.dataframe(pd.DataFrame(st.session_state.products), use_container_width=True)

with t3:
    if st.button('Run Moodboard Composer'):
        dna = st.session_state.style_dna
        products = st.session_state.products
        if not dna or not products:
            st.error('Run Agent 1 and 2 first')
        else:
            s = time.perf_counter()
            board = compose_moodboard(dna, products)
            ms = int((time.perf_counter() - s) * 1000)
            trace('compose_moodboard', True, ms)
            st.session_state.board = board
    if st.session_state.board:
        st.json(st.session_state.board)

with t4:
    q = st.text_input('Ask design question')
    if st.button('Ask'):
        if not api_key:
            st.error('Set OPENAI_API_KEY')
        else:
            dna = st.session_state.style_dna
            products = st.session_state.products
            board = st.session_state.board
            llm = ChatOpenAI(model='gpt-4.1-mini', temperature=0, api_key=api_key)
            answer = llm.invoke(f"Question: {q}\nStyle:{dna}\nProducts:{products[:5]}\nBoard:{board}").content
            st.markdown('**Answer**')
            st.write(answer)
            if auto_qaeval:
                ref = 'Reference should align with current style, products and board context.'
                ev = eval_answer(api_key, q, ref, answer)
                st.caption(f"Auto QAEval: {ev['grade']}")
                st.session_state.qa_log.append({'question': q, 'reference': ref, 'prediction': answer, 'qaeval_grade': ev['grade'], 'qaeval_reason': ev['reason']})

    if st.session_state.qa_log:
        df = pd.DataFrame(st.session_state.qa_log)
        total = len(df)
        correct = int((df['qaeval_grade'] == 'CORRECT').sum())
        st.metric('QAEval Accuracy', f"{(correct/total*100):.2f}%")
        st.dataframe(df, use_container_width=True)

    st.markdown('**MCP Trace Panel**')
    if st.session_state.mcp_trace:
        tr = pd.DataFrame(st.session_state.mcp_trace)
        st.dataframe(tr, use_container_width=True)
