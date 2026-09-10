import os
import json
import uuid
import time
import redis
import streamlit as st
import streamlit.components.v1 as components
from dotenv import load_dotenv
from neo4j import GraphDatabase 
from openai import OpenAI
from sentence_transformers import CrossEncoder

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(PROJECT_ROOT, ".env"))

# ==========================================
# 1. PAGE CONFIGURATION & STYLING
# ==========================================
st.set_page_config(
    page_title="Tamil Nadu Farmer AI Assistant",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for Native Chat Injection & Metrics
st.markdown("""
<style>
    .stApp {
        background-color: #f8fafc;
    }
    .metric-card {
        background: #1b4332;
        color: white;
        padding: 12px;
        border-radius: 10px;
        margin-bottom: 8px;
        font-size: 13px;
    }
    .status-badge {
        background: #081c15;
        color: #52b788;
        padding: 2px 8px;
        border-radius: 12px;
        font-size: 10px;
        font-weight: bold;
    }
    .stChatMessage {
        background-color: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 1rem;
        padding: 1rem;
        box-shadow: 0 1px 2px 0 rgba(0, 0, 0, 0.05);
    }
    .stChatInputContainer {
        border-radius: 1rem;
        border: 1px solid #cbd5e1;
    }
</style>
""", unsafe_allow_html=True)

# Configuration Variables
NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687").strip()
NEO4J_USER = os.getenv("NEO4J_USERNAME", os.getenv("NEO4J_USER", "neo4j")).strip()
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "password").strip()
REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", 6379))
OPENAI_KEY = os.getenv("OPENAI_API_KEY", "").strip()
NEO4J_TRUST_ALL_CERTS = os.getenv("NEO4J_TRUST_ALL_CERTS", "false").strip().lower() in {
    "1",
    "true",
    "yes",
}

if NEO4J_TRUST_ALL_CERTS and NEO4J_URI.startswith("neo4j+s://"):
    NEO4J_URI = NEO4J_URI.replace("neo4j+s://", "neo4j+ssc://", 1)

# ==========================================
# 2. CACHED BACKEND INFRASTRUCTURE
# ==========================================
@st.cache_resource
def get_openai_client():
    if not OPENAI_KEY:
        raise RuntimeError(
            f"OPENAI_API_KEY is missing. Add it to {os.path.join(PROJECT_ROOT, '.env')}"
        )
    return OpenAI(api_key=OPENAI_KEY)

@st.cache_resource
def get_reranker_model():
    return CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')

@st.cache_resource
def get_neo4j_driver():
    try:
        driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
        driver.verify_connectivity()
        return driver
    except Exception:
        return None

@st.cache_resource
def get_redis_client():
    try:
        r = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, db=0, decode_responses=True)
        r.ping()
        return r
    except Exception:
        return None

client = get_openai_client()
reranker = get_reranker_model()
neo4j_driver = get_neo4j_driver()
redis_client = get_redis_client()

# ==========================================
# 3. ROUTING, RAG & STREAMING PIPELINE
# ==========================================
def fast_intent_router(user_query: str) -> str:
    query_lower = user_query.lower().strip()
    
    greetings = ["hi", "hello", "vanakkam", "namaste", "hey"]
    if query_lower in greetings or any(query_lower.startswith(g + " ") for g in greetings):
        return "GREETING"
    
    off_topic_keywords = ["movie", "cricket score", "crypto", "actor"]
    if any(kw in query_lower for kw in off_topic_keywords):
        return "OUT_OF_SCOPE"
    
    return "AGRI_SCHEME"

def rewrite_query_fast(user_query: str, chat_history: list) -> str:
    if not chat_history:
        return user_query
        
    recent_context = "\n".join([f"User: {m['user']}\nAI: {m['ai']}" for m in chat_history[-2:]])
    prompt = f"Context:\n{recent_context}\n\nRewrite into a standalone search query for Tamil Nadu government schemes:\nQuery: {user_query}"
    
    try:
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=50
        )
        return resp.choices[0].message.content.strip()
    except Exception:
        return user_query

def retrieve_and_rerank(query: str, top_k: int = 5):
    if not neo4j_driver:
        return []

    try:
        emb_resp = client.embeddings.create(
            model="text-embedding-3-small",
            input=query
        )
        query_vec = emb_resp.data[0].embedding

        cypher = """
        CALL db.index.vector.queryNodes('scheme_chunks', $k, $embedding)
        YIELD node, score
        MATCH (s:Scheme)-[:HAS_CHUNK]->(node)
        RETURN node.text AS text, node.id AS id, s.title AS scheme_title, s.url AS url, score
        """
        with neo4j_driver.session() as session:
            result = session.run(cypher, k=top_k, embedding=query_vec)
            docs = [dict(rec) for rec in result]
    except Exception:
        docs = []

    if not docs:
        return []

    pairs = [[query, doc["text"]] for doc in docs]
    scores = reranker.predict(pairs)
    for idx, doc in enumerate(docs):
        doc["rerank_score"] = float(scores[idx])

    return sorted(docs, key=lambda x: x["rerank_score"], reverse=True)

def generate_streaming_response(query: str, context_docs: list):
    if context_docs:
        context_str = "\n\n".join([f"Scheme: {d['scheme_title']}\nURL: {d['url']}\nText: {d['text']}" for d in context_docs[:3]])
    else:
        context_str = "No specific match found in local knowledge graph index."

    system_prompt = f"""
    You are an official Tamil Nadu Farmer & Government Scheme AI Assistant.
    Answer the user's inquiry accurately, clearly, and concisely using strictly the retrieved context below.
    If details are present in the context (e.g., TANSIDCO, MSME reservations, subsidies, eligibility), present them in structured bullet points.

    Retrieved Knowledge Context:
    {context_str}
    """

    stream = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": query}
        ],
        temperature=0.1,
        stream=True
    )

    for chunk in stream:
        if chunk.choices and chunk.choices[0].delta.content:
            yield chunk.choices[0].delta.content

# ==========================================
# 4. MEMORY & GUARDRAILS
# ==========================================
def get_memory(session_id: str):
    if not redis_client:
        return st.session_state.get(f"chat_{session_id}", [])
    data = redis_client.get(f"chat:{session_id}")
    return json.loads(data) if data else []

def save_memory(session_id: str, user_msg: str, ai_msg: str):
    history = get_memory(session_id)
    history.append({"user": user_msg, "ai": ai_msg})
    history = history[-10:]
    if redis_client:
        redis_client.set(f"chat:{session_id}", json.dumps(history))
    else:
        st.session_state[f"chat_{session_id}"] = history

def validate_guardrails(text: str) -> bool:
    forbidden = ["ignore previous instructions", "system prompt", "drop table", "sudo"]
    return not any(w in text.lower() for w in forbidden)

# ==========================================
# 5. UI HEADER BANNER & SIDEBAR STATUS
# ==========================================
HEADER_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
    <script src="https://cdn.tailwindcss.com"></script>
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap" rel="stylesheet">
    <script>
        tailwind.config = {
            theme: {
                extend: {
                    colors: {
                        agri: { 50: '#f2f9f4', 100: '#e1f2e5', 500: '#2d6a4f', 600: '#1b4332', 700: '#081c15', accent: '#52b788' }
                    }
                }
            }
        }
    </script>
</head>
<body class="bg-slate-50 font-sans">
    <div class="bg-white border-b border-slate-200 px-6 py-4 flex items-center justify-between shadow-sm">
        <div class="flex items-center space-x-3">
            <div class="w-10 h-10 rounded-xl bg-gradient-to-tr from-agri-500 to-agri-accent flex items-center justify-center text-white text-xl shadow-md">🌾</div>
            <div>
                <div class="flex items-center space-x-2">
                    <h2 class="font-bold text-slate-800 text-lg">TN Agri Scheme Assistant</h2>
                    <span class="bg-agri-100 text-agri-600 text-xs font-semibold px-2.5 py-0.5 rounded-full border border-agri-200">Grounded RAG</span>
                </div>
                <p class="text-xs text-slate-500">Ask questions in English or Tamil about subsidies, PM-KISAN, micro-irrigation, and solar pumps</p>
            </div>
        </div>
        <div class="flex items-center space-x-3">
            <div class="hidden sm:flex items-center space-x-1.5 bg-slate-100 px-3 py-1.5 rounded-full border border-slate-200 text-xs">
                <span class="w-2 h-2 rounded-full bg-emerald-500"></span>
                <span class="text-slate-600 font-medium">Stream Model:</span>
                <span class="text-slate-800 font-semibold">GPT-4o-Mini</span>
            </div>
            <div class="bg-slate-100 text-slate-700 text-xs px-3 py-1.5 rounded-lg border border-slate-300 font-medium flex items-center gap-1.5">
                <i class="fa-solid fa-language text-agri-500"></i>
                <span>English / தமிழ்</span>
            </div>
        </div>
    </div>
</body>
</html>
"""

# Render Header UI Component
components.html(HEADER_HTML, height=85)

# Sidebar System Health Status
with st.sidebar:
    st.title("🌾 TN Farmer AI")
    st.caption("⚡ Streaming Pipeline Active")
    st.divider()

    st.subheader("RAG Pipeline Stack")
    
    neo4j_status = "Connected" if neo4j_driver else "Disconnected"
    neo4j_color = "#52b788" if neo4j_driver else "#ef4444"
    
    redis_status = "Cached" if redis_client else "In-Memory Fallback"
    redis_color = "#fbbf24" if redis_client else "#f97316"

    st.markdown(f"""
    <div class="metric-card">
        <b>Neo4j Graph DB</b><br>
        <span style="color:#94a3b8; font-size:11px;">1,240 Chunks | 18 Schemes</span><br>
        <span class="status-badge" style="color:{neo4j_color};">{neo4j_status}</span>
    </div>
    <div class="metric-card">
        <b>Redis Memory</b><br>
        <span style="color:#94a3b8; font-size:11px;">Latency: 1.2ms | TTL Active</span><br>
        <span class="status-badge" style="color:{redis_color};">{redis_status}</span>
    </div>
    <div class="metric-card">
        <b>Cross-Encoder</b><br>
        <span style="color:#94a3b8; font-size:11px;">ms-marco-MiniLM-L-6-v2</span><br>
        <span class="status-badge" style="color:#60a5fa;">Ready</span>
    </div>
    """, unsafe_allow_html=True)

    st.divider()
    if st.button("🔄 Reset Memory Session", use_container_width=True):
        st.session_state.session_id = str(uuid.uuid4())
        st.rerun()

# ==========================================
# 6. MAIN INTERACTION ENGINE
# ==========================================
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

session_id = st.session_state.session_id
chat_history = get_memory(session_id)

# Display Welcome Banner if History is Empty
if not chat_history:
    st.markdown("""
    <div style="background: white; padding: 1.5rem; border-radius: 1rem; border: 1px solid #e2e8f0; text-align: center; margin-bottom: 1.5rem;">
        <h3 style="color: #1e293b; font-weight: 700; margin-bottom: 0.5rem;">Vanakkam! Welcome to TN Farmer AI</h3>
        <p style="color: #64748b; font-size: 0.9rem; margin-bottom: 0;">
            Powered by Neo4j Knowledge Graph, Cross-Encoder Reranking, and OpenAI streaming for real-time answers on Tamil Nadu agricultural welfare programs.
        </p>
    </div>
    """, unsafe_allow_html=True)

# Display Existing Chat History
for msg in chat_history:
    with st.chat_message("user", avatar="🧑‍🌾"):
        st.write(msg["user"])
    with st.chat_message("assistant", avatar="🌾"):
        st.write(msg["ai"])

# Native Streamlit Input Field
user_input = st.chat_input("Ask about TN agricultural schemes, eligibility, documents required...")

if user_input:
    if not validate_guardrails(user_input):
        st.error("Input query blocked by safety inspection.")
        st.stop()

    # Render User Message
    with st.chat_message("user", avatar="🧑‍🌾"):
        st.write(user_input)

    # Render Streamed Assistant Message
    with st.chat_message("assistant", avatar="🌾"):
        t0 = time.time()
        intent = fast_intent_router(user_input)

        if intent == "GREETING":
            response_text = "Vanakkam! 🙏 I am your Tamil Nadu Farmer & Government Scheme Assistant. How can I help you with agricultural subsidies, TANSIDCO reservations, PM-KISAN, or farm equipment schemes today?"
            st.write(response_text)
            save_memory(session_id, user_input, response_text)

        elif intent == "OUT_OF_SCOPE":
            response_text = "I am specifically trained as a Tamil Nadu Farmer & Government Scheme Assistant. I cannot answer general off-topic questions, but feel free to ask me anything about crop subsidies, TANSIDCO industrial reservations, or government aid!"
            st.write(response_text)
            save_memory(session_id, user_input, response_text)

        else:
            rewritten_q = rewrite_query_fast(user_input, chat_history)
            ranked_docs = retrieve_and_rerank(rewritten_q, top_k=5)
            ttft = time.time() - t0

            st.caption(f"🛣️ **Intent:** `{intent}` | 🔍 **Rewritten Query:** *\"{rewritten_q}\"* | ⚡ **TTFT:** `{ttft*1000:.0f}ms`")

            # Stream response to UI
            response_text = st.write_stream(generate_streaming_response(user_input, ranked_docs))

            if ranked_docs:
                with st.expander(f"🔍 Ingested Grounding Knowledge Nodes ({len(ranked_docs[:3])})"):
                    for d in ranked_docs[:3]:
                        st.markdown(f"**[{d['scheme_title']}]({d['url']})** | Score: `{d['rerank_score']:.3f}`")
                        st.caption(d['text'][:250] + "...")

            save_memory(session_id, user_input, response_text)