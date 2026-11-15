"""
Step 4 - Streamlit UI
Run:  streamlit run app.py
"""
import csv
import datetime as dt

import streamlit as st

from src.config import DATA_DIR, BASE_DIR
from src.ingest import build_vector_store
from src.rag_chain import PolicyAssistant

FEEDBACK_FILE = BASE_DIR / "data" / "feedback.csv"

SAMPLE_QUESTIONS = [
    "How many WFH days can I take per month?",
    "How long is maternity leave?",
    "What is the hotel limit for a metro city?",
    "How long must my password be?",
    "What increment does a rating of 4 get?",
]

st.set_page_config(page_title="PolicySense", page_icon="📘", layout="centered")


# ---------------- helpers ----------------
@st.cache_resource(show_spinner="Loading policies and AI model (first time takes a minute)...")
def load_bot() -> PolicyAssistant:
    return PolicyAssistant()


def save_feedback(question: str, answer: str, value: int) -> None:
    """Append one thumbs up/down row to data/feedback.csv."""
    FEEDBACK_FILE.parent.mkdir(parents=True, exist_ok=True)
    new_file = not FEEDBACK_FILE.exists()
    with open(FEEDBACK_FILE, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if new_file:
            w.writerow(["time", "question", "answer", "rating"])
        w.writerow([dt.datetime.now().isoformat(timespec="seconds"),
                    question, answer, "up" if value == 1 else "down"])


def show_sources(sources: list[dict]) -> None:
    if sources:
        with st.expander(f"Sources ({len(sources)})"):
            for s in sources:
                st.markdown(f"- **{s['source']}**, page {s['page']}")


# ---------------- session state ----------------
if "messages" not in st.session_state:
    st.session_state.messages = []      # {"role", "content", "sources", "question"}
if "rated" not in st.session_state:
    st.session_state.rated = {}         # message index -> rating already saved

# ---------------- sidebar ----------------
with st.sidebar:
    st.title("📘 PolicySense")
    st.caption("AI assistant for employee policy questions")

    st.subheader("Try a question")
    for q in SAMPLE_QUESTIONS:
        if st.button(q, use_container_width=True, key=f"sample_{q}"):
            st.session_state.pending = q

    st.divider()
    if st.button("🗑️ Clear chat", use_container_width=True):
        st.session_state.messages = []
        st.session_state.rated = {}
        st.rerun()

    st.divider()
    st.subheader("Add a policy PDF")
    uploaded = st.file_uploader("Upload a PDF", type=["pdf"])
    if uploaded and st.button("Add to knowledge base", use_container_width=True):
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        (DATA_DIR / uploaded.name).write_bytes(uploaded.getbuffer())
        with st.spinner("Re-indexing all policies..."):
            build_vector_store()
        load_bot.clear()
        st.success(f"Added {uploaded.name}. You can now ask about it.")

    st.divider()
    st.caption("Sample data for demonstration. Answers come only from the uploaded policy documents.")

# ---------------- main area ----------------
st.title("PolicySense")
st.caption("Ask about leave, WFH, travel, security, appraisal and other company policies.")

bot = load_bot()

if not st.session_state.messages:
    st.info("Ask a question below, or pick one from the sidebar.")

# replay the conversation
for i, msg in enumerate(st.session_state.messages):
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant":
            show_sources(msg.get("sources", []))
            rating = st.feedback("thumbs", key=f"fb_{i}")
            if rating is not None and st.session_state.rated.get(i) != rating:
                st.session_state.rated[i] = rating
                save_feedback(msg.get("question", ""), msg["content"], rating)

# new question (typed or from a sidebar button)
prompt = st.chat_input("Ask a policy question...")
if not prompt and "pending" in st.session_state:
    prompt = st.session_state.pop("pending")

if prompt:
    history = [{"role": m["role"], "content": m["content"]}
               for m in st.session_state.messages]

    with st.chat_message("user"):
        st.markdown(prompt)
    with st.chat_message("assistant"):
        with st.spinner("Searching policies..."):
            result = bot.ask(prompt, history)

    st.session_state.messages.append({"role": "user", "content": prompt})
    st.session_state.messages.append({
        "role": "assistant",
        "content": result["answer"],
        "sources": result["sources"],
        "question": prompt,
    })
    st.rerun()