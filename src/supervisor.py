"""
supervisor.py
LangGraph multi-agent supervisor that routes questions between a SQL agent
and a RAG agent, both of which call tools from the MCP server.

Run directly for testing:  python src/supervisor.py
"""

import asyncio
import logging
import sys
import warnings
from pathlib import Path
from typing import Literal, TypedDict

from dotenv import load_dotenv

# Silence noisy warnings so test output is readable
logging.basicConfig(level=logging.CRITICAL, stream=sys.stderr)
logging.getLogger().setLevel(logging.CRITICAL)
warnings.filterwarnings("ignore")

from langchain_groq import ChatGroq
from langchain_core.messages import (
    AIMessage, BaseMessage, HumanMessage, SystemMessage
)
from langchain_core.prompts import ChatPromptTemplate
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import create_react_agent
from pydantic import BaseModel, Field
import asyncio as _asyncio
from typing import Annotated
from operator import add

load_dotenv()

PROJECT_ROOT = Path(__file__).parent.parent
MCP_SERVER   = PROJECT_ROOT / "src" / "mcp_server.py"

# 1. Shared LLM
llm = ChatGroq(model="openai/gpt-oss-120b", temperature=0)

# 2. State definition — flows through every node
class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add]
    next:     str
    called:   Annotated[list[str], add]

# 3. Supervisor prompt
class RoutingDecision(BaseModel):
    """Schema the supervisor LLM MUST return."""
    next: Literal["sql_agent", "rag_agent", "FINISH"] = Field(
        ...,
        description=(
            "Which specialist should handle the NEXT step. "
            "Use 'sql_agent' for questions about customers, tickets, or orders. "
            "Use 'rag_agent' for questions about company policies (returns, refunds, warranty, AppleCare+, etc.). "
            "Use 'FINISH' ONLY when the user's full question has been completely answered by the specialists."
        ),
    )
    reasoning: str = Field(
        ...,
        description="One sentence explaining the routing choice.",
    )

SUPERVISOR_SYSTEM = """You are a routing supervisor for a customer support multi-agent system.

You do NOT talk to the user. You do NOT answer questions. Your ONLY job is to decide which specialist should handle the next step.

Available specialists:
- sql_agent: answers questions about customers, support tickets, and orders (structured database).
- rag_agent: answers questions about company policies — returns, refunds, warranty, AppleCare+, cancellations, exchanges — using PDF knowledge base.

Decision logic:
1. Examine the USER's original question (the first human message).
2. Identify which parts of that question require policy information (rag_agent) and which parts require customer/ticket/order data (sql_agent).
3. Look at which specialists have already answered.
4. If any part of the user's question is STILL UNANSWERED, route to the specialist that can answer it.
5. If ALL parts of the user's question are now answered, choose FINISH.

Compound example:
  User asks: "What's our refund policy, and does customer #5 have open tickets?"
  Turn 1: You choose rag_agent (policy part).
  Turn 2 (after rag_agent answers): You choose sql_agent (customer part still unanswered).
  Turn 3 (after sql_agent answers): You choose FINISH.

Never route to the same specialist twice unless the user explicitly asked for more.
"""

supervisor_prompt = ChatPromptTemplate.from_messages([
    ("system", SUPERVISOR_SYSTEM),
    ("placeholder", "{messages}"),
])

# Bind the LLM to the structured output schema
_supervisor_llm = llm.with_structured_output(RoutingDecision)

def supervisor_node(state: AgentState) -> dict:
    """Hybrid router with verbose debugging."""
    called = state.get("called", [])
    
    print(f"[supervisor DEBUG] called list: {called}", file=sys.stderr)

    # Once both specialists have answered, we're done
    if "sql_agent" in called and "rag_agent" in called:
        print("[supervisor] → FINISH (both specialists answered)", file=sys.stderr)
        return {"next": "FINISH"}

    user_q = next(
        (m.content for m in state["messages"] if isinstance(m, HumanMessage)),
        "",
    ).lower()
    
    print(f"[supervisor DEBUG] user_q: {user_q[:100]}", file=sys.stderr)

    policy_keywords = [
        "policy", "return", "refund", "warranty", "applecare",
        "cancel", "exchange", "store credit", "defective",
        "gift card", "how many days", "days to return", "window",
        "service fee", "screen damage", "back glass", "accidental damage",
    ]
    sql_keywords = [
        "customer #", "customer id", "profile", "ticket", "order",
        "tickets", "orders", "email", "signup", "plan",
        "status", "priority", "urgent", "open tickets", "closed",
        "how many customers", "how many tickets", "list customers",
        "which customers", "show customers", "who has",
        "customers", "how many",
    ]

    needs_policy = any(k in user_q for k in policy_keywords)
    needs_sql    = any(k in user_q for k in sql_keywords)
    
    print(f"[supervisor DEBUG] needs_policy={needs_policy}, needs_sql={needs_sql}", file=sys.stderr)
    
    # Compound question — needs both agents
    if needs_policy and needs_sql:
        if "rag_agent" not in called:
            print("[supervisor] → rag_agent (compound: policy first)", file=sys.stderr)
            return {"next": "rag_agent"}
        if "sql_agent" not in called:
            print("[supervisor] → sql_agent (compound: sql second)", file=sys.stderr)
            return {"next": "sql_agent"}
        print("[supervisor] → FINISH (compound both answered)", file=sys.stderr)
        return {"next": "FINISH"}

    # Pure policy question
    if needs_policy and not needs_sql:
        if "rag_agent" not in called:
            print("[supervisor] → rag_agent (policy-only)", file=sys.stderr)
            return {"next": "rag_agent"}
        print("[supervisor] → FINISH (policy-only answered)", file=sys.stderr)
        return {"next": "FINISH"}

    # Pure SQL question
    if needs_sql and not needs_policy:
        if "sql_agent" not in called:
            print("[supervisor] → sql_agent (data-only)", file=sys.stderr)
            return {"next": "sql_agent"}
        print("[supervisor] → FINISH (data-only answered)", file=sys.stderr)
        return {"next": "FINISH"}

    # Truly ambiguous — no keywords matched
    # If any specialist has already answered, finish. Don't use LLM.
    if called:
        print(f"[supervisor] → FINISH (ambiguous, but {called} already answered)", file=sys.stderr)
        return {"next": "FINISH"}

    # Genuinely new ambiguous question with nothing answered yet. Use LLM
    try:
        context = f"User's question: {user_q}\n\nDecide: sql_agent, rag_agent, or FINISH."
        decision: RoutingDecision = _supervisor_llm.invoke(
            [SystemMessage(content=SUPERVISOR_SYSTEM),
             HumanMessage(content=context)]
        )
        print(f"[supervisor] → {decision.next} (LLM fallback: {decision.reasoning})", file=sys.stderr)
        return {"next": decision.next}
    except Exception as e:
        print(f"[supervisor error: {e}]", file=sys.stderr)
        return {"next": "FINISH"}

# 4. Worker agents: each is a ReAct agent with MCP tools
SQL_SYSTEM = """You are a SQL specialist for a customer support team.

You have tools to query a SQLite database containing:
- customers (customer_id, name, email, plan, signup_date, country)
- tickets (ticket_id, customer_id, subject, description, status, priority, created_at, resolution)
- orders (order_id, customer_id, product, amount_usd, order_date, status)

Rules:
1. Use the provided tools to answer. Never guess.
2. If the user mentions a customer by name, call query_customer_profile first to get their customer_id, then use that ID for tickets/orders lookups.
3. For anything complex that the dedicated tools can't express, use run_sql with a SELECT query.
4. Summarize results in plain English for a non-technical user.
5. If you cannot answer from this database, say so clearly — do not make up data.
"""

RAG_SYSTEM = """You are a company policy specialist.

You have a tool (search_policy_documents) that searches company policy PDFs using semantic similarity.

Rules:
1. ALWAYS call search_policy_documents before answering. Never answer from memory.
2. Base your answer ONLY on the retrieved passages.
3. Cite the source file for every fact, formatted as "(Source: filename.pdf)".
4. Quote exact policy language for key facts (timeframes, fees, conditions).
5. If the retrieved passages don't contain the answer, say:
   "I couldn't find that information in the available policy documents."
6. Keep answers concise and factual.
"""


# 5. The main async build function: sets up MCP, agents, and graph
async def build_graph():
    # Connect to the MCP server, pulling all available tools
    client = MultiServerMCPClient({
        "support": {
            "command": "python",
            "args": [str(MCP_SERVER)],
            "transport": "stdio",
        }
    })
    all_tools = await client.get_tools()

    # Split tools by which agent should use them
    sql_tool_names = {"query_customer_profile", "query_tickets",
                      "query_orders", "run_sql"}
    rag_tool_names = {"search_policy_documents"}

    sql_tools = [t for t in all_tools if t.name in sql_tool_names]
    rag_tools = [t for t in all_tools if t.name in rag_tool_names]

    # Build the two worker agents as ReAct agents
    sql_agent = create_react_agent(model=llm, tools=sql_tools, prompt=SQL_SYSTEM)
    rag_agent = create_react_agent(model=llm, tools=rag_tools, prompt=RAG_SYSTEM)

    # Node wrappers: each takes/returns AgentState
    async def sql_node(state: AgentState) -> dict:
        user_msgs = [m for m in state["messages"] if isinstance(m, HumanMessage)]
        result = await _retry_on_ratelimit(
            lambda: sql_agent.ainvoke({"messages": user_msgs})
        )
        last = result["messages"][-1]
        return {
            "messages": [AIMessage(content=last.content, name="sql_agent")],
            "called":   state.get("called", []) + ["sql_agent"],
        }

    async def rag_node(state: AgentState) -> dict:
        user_msgs = [m for m in state["messages"] if isinstance(m, HumanMessage)]
        result = await _retry_on_ratelimit(
            lambda: rag_agent.ainvoke({"messages": user_msgs})
        )
        last = result["messages"][-1]
        return {
            "messages": [AIMessage(content=last.content, name="rag_agent")],
            "called":   state.get("called", []) + ["rag_agent"],
        }

    # Build the graph
    graph = StateGraph(AgentState)
    graph.add_node("supervisor", supervisor_node)
    graph.add_node("sql_agent", sql_node)
    graph.add_node("rag_agent", rag_node)

    # From START, go to supervisor first
    graph.add_edge(START, "supervisor")

    # Conditional edge from supervisor: read state["next"] to decide
    def route_from_supervisor(state: AgentState) -> Literal["sql_agent", "rag_agent", "__end__"]:
        if state["next"] == "FINISH":
            return "__end__"
        return state["next"]

    graph.add_conditional_edges("supervisor", route_from_supervisor)

    # After each worker finishes, control returns to the supervisor
    graph.add_edge("sql_agent", "supervisor")
    graph.add_edge("rag_agent", "supervisor")

    return graph.compile()


# 6. Convenience: ask() function for tests and UI
_compiled_graph = None

async def _get_graph():
    global _compiled_graph
    if _compiled_graph is None:
        _compiled_graph = await build_graph()
    return _compiled_graph

async def ask_async(question: str) -> dict:
    """Returns a dict: {"answer": str, "agents": list[str]}"""
    g = await _get_graph()
    initial = {
        "messages": [HumanMessage(content=question)],
        "next":     "",
        "called":   [],
    }
    final = await g.ainvoke(initial, config={"recursion_limit": 12})

    agents_used = []
    answers = []
    for m in final["messages"]:
        if isinstance(m, AIMessage) and getattr(m, "name", None) in ("sql_agent", "rag_agent"):
            if m.name not in agents_used:
                agents_used.append(m.name)
            answers.append(m.content)

    if not answers:
        return {"answer": "(no answer produced)", "agents": []}
    if len(answers) == 1:
        return {"answer": answers[0], "agents": agents_used}
    return {"answer": "\n\n---\n\n".join(answers), "agents": agents_used}

def ask(question: str) -> str:
    result = asyncio.run(ask_async(question))
    tag = f"[Handled by: {', '.join(result['agents'])}]\n\n" if result['agents'] else ""
    return tag + result['answer']

import asyncio as _asyncio

async def _retry_on_ratelimit(coro_fn, max_retries=3):
    """Run an async callable, retrying on Groq 429 errors."""
    for attempt in range(max_retries):
        try:
            return await coro_fn()
        except Exception as e:
            if "rate_limit" in str(e).lower() or "429" in str(e):
                wait = 2 ** attempt * 5   # 5s, 10s, 20s
                print(f"[rate limited, sleeping {wait}s before retry {attempt+1}/{max_retries}]", file=sys.stderr)
                await _asyncio.sleep(wait)
            else:
                raise
    raise RuntimeError("Exceeded rate-limit retries")


# 7. Test
if __name__ == "__main__":
    import time

    test_questions = [
        "How many customers do we have in total?",
        "What is Apple's return window for items purchased online?",
        "Give me a profile for customer #1 and list their recent tickets.",
        "What is the AppleCare+ service fee for iPhone Screen-Only Damage?",
        "What is our refund policy for defective items, and does customer #1 have any open tickets about defective products?",
    ]

    for i, q in enumerate(test_questions):
        print("\n" + "=" * 72)
        print(f"Q: {q}")
        print("-" * 72)
        try:
            print(ask(q))
        except Exception as e:
            print(f"ERROR: {e}")
        # Pause between questions to respect Groq's per-minute token budget
        if i < len(test_questions) - 1:
            time.sleep(8)