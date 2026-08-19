# Aria — Autonomous AI Support Agent

An **agentic AI system** that reads an incoming customer support message and
autonomously decides, step by step, how to handle it — instead of just
generating a reply.

## What it does

1. A customer describes an issue in a chat window.
2. The agent (Llama 3.3 70B, via the Groq API) reasons about the problem and
   **decides on its own** whether to:
   - search the internal knowledge base for a known fix, and answer directly, or
   - escalate the issue by **creating a support ticket** with the correct
     category, priority, and SLA.
3. Every tool call the agent makes is shown live in the UI, so you can see
   its reasoning trace, not just the final answer.

## Why this project

Built to demonstrate practical, hands-on understanding of:

- **Agentic AI** — LLM-driven autonomous decision loops (reason → act → observe → repeat)
- **Tool / function calling** — giving an LLM real functions it can invoke
- **RAG-style retrieval** — grounding answers in a knowledge base instead of hallucinating
- **LLM APIs** — Groq's free, fast inference API (Llama 3.3 70B)
- **Applying AI to a real support/ITSM workflow** (ticket triage, priority, SLA)

## Tech stack

- Python
- Streamlit (UI)
- Groq API (LLM + tool calling)
- scikit-learn (TF-IDF search over the knowledge base)

## Running locally

```bash
pip install -r requirements.txt
cp .env.example .env   # then paste your Groq API key into .env
streamlit run app.py
```

## Deployment

Deployed for free on Streamlit Community Cloud. Set `GROQ_API_KEY` as a
secret in the app settings (see project write-up for full steps).

## Project structure

```
support-agent-ai/
├── app.py            # Streamlit UI (chat + live agent reasoning panel)
├── agent.py           # Agent orchestration logic + tools (the "brain")
├── kb.py               # Knowledge base search (TF-IDF)
├── kb_data/articles.json  # Sample support knowledge base
├── requirements.txt
└── .env.example
```
