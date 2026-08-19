import os
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

from agent import run_agent, TICKETS

st.set_page_config(page_title="Aria - AI Support Agent", page_icon="🤖", layout="wide")

st.title("🤖 Aria — Autonomous AI Support Agent")
st.caption(
    "An agentic AI system that reads a customer's issue, decides on its own whether "
    "to solve it using the knowledge base or escalate it as a support ticket — "
    "powered by Llama 3.3 (via Groq)."
)

if "GROQ_API_KEY" not in os.environ or not os.environ.get("GROQ_API_KEY"):
    st.error(
        "GROQ_API_KEY not found. Add it as an environment variable / Streamlit secret "
        "before using the app (see README.md)."
    )
    st.stop()

if "messages" not in st.session_state:
    st.session_state.messages = []
if "trace_log" not in st.session_state:
    st.session_state.trace_log = []

left, right = st.columns([2, 1])

with left:
    st.subheader("💬 Chat with the support agent")

    for m in st.session_state.messages:
        with st.chat_message(m["role"]):
            st.markdown(m["content"])

    user_input = st.chat_input("Describe your issue, e.g. 'I can't log into my account'")

    if user_input:
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        with st.chat_message("assistant"):
            with st.spinner("Agent is thinking..."):
                reply, trace = run_agent(user_input, st.session_state.messages[:-1])
                st.markdown(reply)

        st.session_state.messages.append({"role": "assistant", "content": reply})
        st.session_state.trace_log.append({"user_message": user_input, "trace": trace})
        st.rerun()

with right:
    st.subheader("🧠 Agent reasoning (live)")
    st.caption("This shows exactly what tools the agent chose to call and why — proof it's reasoning, not just replying.")

    if not st.session_state.trace_log:
        st.info("Ask something in the chat to see the agent's decision process here.")
    else:
        for entry in reversed(st.session_state.trace_log):
            with st.expander(f"🗨️ \"{entry['user_message'][:40]}...\"", expanded=(entry is st.session_state.trace_log[-1])):
                for step in entry["trace"]:
                    if step["step"] == "tool_call":
                        st.markdown(f"**🔧 Called tool:** `{step['tool']}`")
                        st.json(step["args"])
                        st.markdown("**↳ Result:**")
                        st.json(step["result"])
                    else:
                        st.markdown("**✅ Gave final answer to customer.**")

    st.divider()
    st.subheader("🎫 Tickets raised so far")
    if not TICKETS:
        st.caption("No tickets escalated yet.")
    else:
        for t in reversed(TICKETS):
            st.markdown(
                f"**{t['ticket_id']}** · {t['priority']} priority · {t['category']}  \n"
                f"{t['summary']}  \n"
                f"SLA: {t['sla_hours']}h · Status: {t['status']}"
            )
            st.divider()

st.sidebar.header("About this project")
st.sidebar.markdown(
    """
This project demonstrates:
- **Agentic AI** — the LLM autonomously decides which tools to call and when
- **Tool / function calling** — real Python functions the agent can trigger
- **RAG-style knowledge retrieval** — searches a support knowledge base
- **Simulated ticketing system** — escalates unresolved issues automatically
- **LLM** — Llama 3.3 70B served for free via the Groq API

Built with Python + Streamlit + Groq.
"""
)

if st.sidebar.button("🔄 Reset conversation"):
    st.session_state.messages = []
    st.session_state.trace_log = []
    st.rerun()
