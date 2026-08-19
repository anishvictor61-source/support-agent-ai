"""
agent.py
---------
This is the "brain" of the project: an autonomous AI Support Agent.

WHAT MAKES THIS "AGENTIC" (not just a chatbot):
Instead of the LLM only replying with text, it is given a set of TOOLS
(real functions in this file) and is allowed to DECIDE on its own,
step by step:
  1. Should I search the knowledge base?
  2. Do I have enough info to resolve this myself?
  3. Should I escalate this and raise a support ticket instead?
  4. What priority does this deserve?

The LLM calls these tools itself (this is called "function calling" /
"tool use"), we run the real Python function, feed the result back to
the LLM, and it keeps reasoning until it gives a final answer. This
loop is the core pattern behind every "AI agent" product you hear about.

Model: Llama 3.3 70B via Groq (free, fast API)
"""

import json
import os
import uuid
from datetime import datetime

from groq import Groq
from kb import KnowledgeBase

MODEL = "llama-3.3-70b-versatile"

kb = KnowledgeBase()

# In-memory "ticket system" - stands in for a real system like Jira/Zendesk.
# In a real company this function would call the Jira/Zendesk API instead.
TICKETS = []


# ---------------------------------------------------------------------
# TOOLS the agent is allowed to use
# ---------------------------------------------------------------------

def search_knowledge_base(query: str) -> str:
    results = kb.search(query, top_k=2)
    if not results:
        return json.dumps({"found": False, "message": "No relevant KB article found."})
    return json.dumps({"found": True, "articles": results})


def create_support_ticket(summary: str, category: str, priority: str) -> str:
    ticket_id = f"TCKT-{str(uuid.uuid4())[:8].upper()}"
    sla_hours = {"Low": 48, "Medium": 24, "High": 4, "Critical": 1}.get(priority, 24)
    ticket = {
        "ticket_id": ticket_id,
        "summary": summary,
        "category": category,
        "priority": priority,
        "sla_hours": sla_hours,
        "created_at": datetime.utcnow().isoformat() + "Z",
        "status": "Open",
    }
    TICKETS.append(ticket)
    return json.dumps(ticket)


TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search_knowledge_base",
            "description": "Search the internal support knowledge base for articles relevant to the customer's issue. Always try this before giving up on solving something yourself.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "A short search query describing the customer's problem.",
                    }
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "create_support_ticket",
            "description": "Escalate the issue by creating a support ticket for a human agent. Use this ONLY when you cannot resolve the issue yourself from the knowledge base, or when the issue is high/critical priority (e.g. outage, data loss, security, billing dispute).",
            "parameters": {
                "type": "object",
                "properties": {
                    "summary": {"type": "string", "description": "One-line summary of the issue."},
                    "category": {
                        "type": "string",
                        "enum": [
                            "Account & Login",
                            "Integrations",
                            "Billing",
                            "Performance",
                            "Data & Exports",
                            "Other",
                        ],
                    },
                    "priority": {
                        "type": "string",
                        "enum": ["Low", "Medium", "High", "Critical"],
                    },
                },
                "required": ["summary", "category", "priority"],
            },
        },
    },
]

AVAILABLE_FUNCTIONS = {
    "search_knowledge_base": search_knowledge_base,
    "create_support_ticket": create_support_ticket,
}

SYSTEM_PROMPT = """You are Aria, an autonomous AI Support Agent for a SaaS company.

Your job for every customer message:
1. Understand the issue.
2. Use the search_knowledge_base tool to look for a relevant solution before answering.
3. If the knowledge base gives you a clear fix, explain it to the customer yourself,
   in a friendly, simple, step-by-step way. Do NOT escalate if you can solve it.
4. If the knowledge base has no good answer, OR the issue sounds high priority
   (outage, data loss, security, billing dispute, many users affected), use the
   create_support_ticket tool to escalate it, and tell the customer their ticket
   number and expected response time.
5. Always be concise, clear, and reassuring. Never invent information that isn't
   in the knowledge base or obvious common sense.
"""


def run_agent(user_message: str, history: list):
    """
    Runs one full agent turn. Returns:
        final_reply: str
        trace: list of steps the agent took (for showing "agent thinking" in the UI)
    """
    client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    messages.extend(history)
    messages.append({"role": "user", "content": user_message})

    trace = []
    max_steps = 5

    for step in range(max_steps):
        response = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            tools=TOOLS,
            tool_choice="auto",
            temperature=0.3,
        )
        msg = response.choices[0].message

        # No more tool calls -> agent has its final answer
        if not msg.tool_calls:
            trace.append({"step": "final_answer", "content": msg.content})
            return msg.content, trace

        # Agent decided to call one or more tools
        messages.append(
            {
                "role": "assistant",
                "content": msg.content or "",
                "tool_calls": [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments,
                        },
                    }
                    for tc in msg.tool_calls
                ],
            }
        )

        for tool_call in msg.tool_calls:
            fn_name = tool_call.function.name
            fn_args = json.loads(tool_call.function.arguments)
            fn = AVAILABLE_FUNCTIONS.get(fn_name)
            result = fn(**fn_args) if fn else json.dumps({"error": "unknown tool"})

            trace.append({"step": "tool_call", "tool": fn_name, "args": fn_args, "result": json.loads(result)})

            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "name": fn_name,
                    "content": result,
                }
            )

    # Safety fallback if the loop somehow doesn't converge
    return "I've escalated this to a human agent for further review.", trace
