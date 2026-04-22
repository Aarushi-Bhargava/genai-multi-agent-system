"""
sql_agent.py
A standalone SQL agent that answers natural-language questions
about the customers database.

Run directly for testing: python src/sql_agent.py
"""

import os
from pathlib import Path
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits.sql.toolkit import SQLDatabaseToolkit
from langgraph.prebuilt import create_react_agent

# 1. Load environment variables (Groq API key)
load_dotenv()

# 2. Connect to the SQLite database we built in Stage 2
DB_PATH = Path(__file__).parent.parent / "data" / "customers.db"
db = SQLDatabase.from_uri(f"sqlite:///{DB_PATH}")

# 3. Initializing the LLM
llm = ChatGroq(
    model="openai/gpt-oss-120b",
    temperature=0,
)

# 4. Build the SQL toolkit — a bundle of tools the agent can use:
#    - list tables
#    - get schema of a table
#    - run a SQL query
#    - check a query before running it
toolkit = SQLDatabaseToolkit(db=db, llm=llm)
tools = toolkit.get_tools()

# 5. Write a clear system prompt so the agent knows its job
SYSTEM_PROMPT = """You are a SQL expert assisting a customer support team.

You have access to a SQLite database with three tables:
- customers (customer_id, name, email, plan, signup_date, country)
- tickets (ticket_id, customer_id, subject, description, status, priority, created_at, resolution)
- orders (order_id, customer_id, product, amount_usd, order_date, status)

Rules:
1. ALWAYS inspect the schema first using the available tools before writing queries.
2. Only use SELECT queries. Never INSERT, UPDATE, DELETE, or DROP.
3. Limit results to at most 10 rows unless the user asks for more.
4. If a customer's name is ambiguous, search with LIKE and show matches.
5. After getting results, summarize them in plain English for a non-technical user.
6. If you cannot answer the question from this database, say so clearly.
7. When describing results, use the exact priority/status values as they appear in the database. Do not equate 'high' and 'urgent' — they are distinct priority levels.
"""

# 6. Create the ReAct (reason + act) agent
sql_agent = create_react_agent(
    model=llm,
    tools=tools,
    prompt=SYSTEM_PROMPT,
)

# 7. Helper function to ask the agent a question
def ask(question: str) -> str:
    """Send a natural-language question to the SQL agent, return its answer."""
    try:
        result = sql_agent.invoke({"messages": [("user", question)]})
        return result["messages"][-1].content
    except Exception as e:
        return f"Sorry, I hit an error while processing that question: {e}"


# 8. Test
if __name__ == "__main__":
    print("SQL Agent ready. Type a question (or 'quit' to exit):\n")
    while True:
        q = input("You: ").strip()
        if q.lower() in ("quit", "exit", ""):
            break
        print(f"\nAgent: {ask(q)}\n")