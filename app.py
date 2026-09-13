import json
import os
from pathlib import Path

import numpy as np
import streamlit as st
from sentence_transformers import SentenceTransformer
from google import genai

st.set_page_config(
    page_title="FreshBus Assistant",
    page_icon="🚌",
    layout="centered",
)

DATA_PATH = Path(__file__).parent / "data" / "knowledge_base.json"
MODEL_NAME = "all-MiniLM-L6-v2"
LLM_MODEL = "gemini-2.5-flash"

@st.cache_data
def load_knowledge():
    return json.loads(DATA_PATH.read_text(encoding="utf-8"))

@st.cache_resource
def load_embedder():
    return SentenceTransformer(MODEL_NAME)

@st.cache_data
def build_index(texts):
    model = load_embedder()
    embeddings = model.encode(
        texts,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    return np.asarray(embeddings, dtype=np.float32)

def retrieve(question, records, embeddings, top_k=3, threshold=0.28):
    model = load_embedder()
    q = model.encode([question], normalize_embeddings=True)[0]
    scores = embeddings @ q
    order = np.argsort(scores)[::-1][:top_k]
    results = []
    for idx in order:
        score = float(scores[idx])
        if score >= threshold:
            item = dict(records[idx])
            item["score"] = score
            results.append(item)
    return results

def answer_with_gemini(question, history, context):
    api_key = st.secrets.get("GEMINI_API_KEY", os.getenv("GEMINI_API_KEY"))
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY is not configured.")

    client = genai.Client(api_key=api_key)

    context_text = "\n\n".join(
        f"[Source: {x['source']} | Category: {x['category']}]\n{x['text']}"
        for x in context
    )

    history_text = "\n".join(
        f"{m['role'].upper()}: {m['content']}" for m in history[-6:]
    )

    prompt = f"""
You are FreshBus Assistant, a customer support chatbot.

Answer the user's question using ONLY the FreshBus information in CONTEXT.
Do not use general knowledge to invent FreshBus policies, prices, routes,
refund rules, product features, or operational procedures.

If the context does not contain enough information to answer confidently,
say that you could not find the relevant FreshBus information in the
available knowledge base and recommend contacting FreshBus support.

If the question asks for current/live information such as live bus location,
real-time seat availability, or a booking transaction, explain that this
prototype does not have access to live operational systems.

Keep the answer concise and helpful. When useful, mention the relevant
source category.

CONTEXT:
{context_text}

RECENT CONVERSATION:
{history_text}

USER QUESTION:
{question}
"""
    response = client.models.generate_content(
        model=LLM_MODEL,
        contents=prompt,
    )
    return response.text

records = load_knowledge()
texts = [x["text"] for x in records]
embeddings = build_index(texts)

st.title("🚌 FreshBus Assistant")
st.caption("A simple RAG-based customer support prototype using publicly available FreshBus information.")

if "messages" not in st.session_state:
    st.session_state.messages = []

with st.sidebar:
    st.subheader("Prototype scope")
    st.write(
        "Answers are grounded in a limited FreshBus knowledge base. "
        "Live booking, tracking, seat availability and account actions are not connected."
    )
    st.divider()
    st.write(f"Knowledge items: {len(records)}")
    st.write(f"Retrieval: top 3 semantic matches")
    st.write("LLM: Gemini 2.5 Flash")

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message.get("sources"):
            st.caption("Sources: " + " • ".join(message["sources"]))

question = st.chat_input("Ask about FreshBus bookings, refunds, Green Coins, Fresh Card, Fresh Promise, or group bookings")

if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    retrieved = retrieve(question, records, embeddings)

    with st.chat_message("assistant"):
        if not retrieved:
            response = (
                "I couldn't find enough relevant FreshBus information in my "
                "available knowledge base to answer that confidently. "
                "Please contact FreshBus support for confirmation."
            )
            sources = []
        else:
            try:
                response = answer_with_gemini(
                    question,
                    st.session_state.messages[:-1],
                    retrieved,
                )
            except Exception as exc:
                response = f"Sorry, the AI service could not respond right now. Please try again. ({type(exc).__name__})"
            sources = list(dict.fromkeys(x["source"] for x in retrieved))

        st.markdown(response)
        if sources:
            st.caption("Retrieved from: " + " • ".join(sources))

    st.session_state.messages.append(
        {"role": "assistant", "content": response, "sources": sources}
    )
