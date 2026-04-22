"""
app.py
Streamlit UI for the multi-agent customer support system.

Run:  streamlit run src/app.py
"""
import asyncio
import logging
import shutil
import sys
import warnings
from pathlib import Path

import streamlit as st

# Silence noisy warnings
logging.basicConfig(level=logging.CRITICAL, stream=sys.stderr)
logging.getLogger().setLevel(logging.CRITICAL)
warnings.filterwarnings("ignore")

# Import supervisor (MCP + embeddings loading)
from supervisor import ask_async, _get_graph

# Paths
PROJECT_ROOT = Path(__file__).parent.parent
POLICIES_DIR = PROJECT_ROOT / "data" / "policies"
VECTOR_DIR   = PROJECT_ROOT / "data" / "chroma_db"

# Page configuration
st.set_page_config(
    page_title="Customer Support Assistant",
    page_icon="🎧",
    layout="wide",
)

# Sidebar: PDF upload + system info
with st.sidebar:
    st.title("🎧 Support Assistant")
    st.markdown(
        "A multi-agent system that answers questions about "
        "**customer data** and **company policies** using "
        "LangGraph, MCP, and Groq."
    )
    st.divider()

    st.subheader("📄 Knowledge Base")
    existing_pdfs = sorted(POLICIES_DIR.glob("*.pdf"))
    if existing_pdfs:
        st.caption("Currently indexed:")
        for p in existing_pdfs:
            st.text(f"• {p.name}")
    else:
        st.warning("No PDFs indexed yet.")

    st.divider()
    st.subheader("📤 Upload New Policy PDF")
    uploaded = st.file_uploader(
        "Add a PDF to the knowledge base",
        type=["pdf"],
        accept_multiple_files=False,
    )
    if uploaded is not None:
        dest = POLICIES_DIR / uploaded.name
        POLICIES_DIR.mkdir(parents=True, exist_ok=True)
        with open(dest, "wb") as f:
            f.write(uploaded.getbuffer())
        st.success(f"Saved {uploaded.name}")
        st.info(
            "Run `python src/ingest.py` in your terminal to re-index the "
            "knowledge base, then restart this app."
        )

    st.divider()
    st.subheader("🏗 Architecture")
    st.markdown(
        "- **Supervisor** (LangGraph) routes between specialists\n"
        "- **SQL Agent** — customer/ticket/order lookups\n"
        "- **RAG Agent** — policy search over PDFs\n"
        "- **MCP Server** exposes tools to both agents\n"
        "- **Groq** (`gpt-oss-120b`) as the LLM"
    )

    if st.button("🗑 Clear chat"):
        st.session_state.messages = []
        st.rerun()

# Main chat area
st.title("Customer Support Assistant")
st.caption(
    "Ask about customers, tickets, orders, or company policies (returns, "
    "refunds, warranty, AppleCare+)."
)

# Initialize chat history in session state
if "messages" not in st.session_state:
    st.session_state.messages = []

# Render previous chat turns
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Chat input
user_input = st.chat_input("Ask a question…")

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        placeholder = st.empty()
        placeholder.markdown("⏳ Routing your question through the multi-agent system…")

        try:
            result = asyncio.run(ask_async(user_input))
            agents = result["agents"]
            answer = result["answer"]
        except Exception as e:
            agents = []
            answer = f"⚠️ I hit an error: {e}"

        # Show which agents handled the question as a badge
        if agents:
            agent_label = " + ".join(a.replace("_", " ").title() for a in agents)
            badge = f"🤖 *Handled by: **{agent_label}***"
            full_response = f"{badge}\n\n{answer}"
        else:
            full_response = answer

        placeholder.markdown(full_response)

    st.session_state.messages.append({"role": "assistant", "content": full_response})