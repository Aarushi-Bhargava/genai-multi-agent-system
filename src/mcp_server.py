"""
mcp_server.py
An MCP (Model Context Protocol) server that exposes tools for:
  - Querying the customer SQLite database
  - Searching the policy PDFs via the ChromaDB vector store

Run directly:  python src/mcp_server.py

The server communicates over stdio, following the MCP standard.
Any MCP-compliant client (our agents, Claude Desktop, etc.) can use it.
"""

import logging
import sys
import warnings

# Silence all library logging/warnings on stdout so they don't corrupt
# the MCP JSON-RPC protocol on stdio.
logging.basicConfig(level=logging.CRITICAL, stream=sys.stderr)
logging.getLogger().setLevel(logging.CRITICAL)
warnings.filterwarnings("ignore")

# Existing imports below...
import sqlite3
from pathlib import Path
from typing import Optional

from mcp.server.fastmcp import FastMCP
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

# Paths
PROJECT_ROOT = Path(__file__).parent.parent
DB_PATH      = PROJECT_ROOT / "data" / "customers.db"
VECTOR_DIR   = PROJECT_ROOT / "data" / "chroma_db"

# Initialize the MCP server.
mcp = FastMCP("customer-support-tools")

# Initialize the vector store ONCE, when the server starts (Aiming to avoid having to reload the embedding model on every request.)
_embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True},
)
_vectorstore = Chroma(
    persist_directory=str(VECTOR_DIR),
    embedding_function=_embeddings,
)

# TOOL 1: Query customers by name or email
@mcp.tool()
def query_customer_profile(name_or_email: str) -> str:
    """Look up a customer's profile by partial name or email match.

    Args:
        name_or_email: A substring of the customer's name or email address.

    Returns:
        Formatted customer profile(s), or a message if no match found.
    """
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()
    cur.execute(
        """
        SELECT customer_id, name, email, plan, signup_date, country
        FROM customers
        WHERE name LIKE ? OR email LIKE ?
        LIMIT 5
        """,
        (f"%{name_or_email}%", f"%{name_or_email}%"),
    )
    rows = cur.fetchall()
    conn.close()

    if not rows:
        return f"No customers found matching '{name_or_email}'."

    out = []
    for r in rows:
        out.append(
            f"Customer #{r['customer_id']}: {r['name']}\n"
            f"  Email: {r['email']}\n"
            f"  Segment: {r['plan']}\n"
            f"  Country: {r['country']}\n"
            f"  Signed up: {r['signup_date']}"
        )
    return "\n\n".join(out)


# TOOL 2: Query a customer's tickets
@mcp.tool()
def query_tickets(
    customer_id: Optional[int] = None,
    status: Optional[str] = None,
    priority: Optional[str] = None,
    limit: int = 10,
) -> str:
    """Fetch support tickets, optionally filtered by customer, status, or priority.

    Args:
        customer_id: If provided, only tickets for this customer are returned.
        status: Filter by status ('open', 'in_progress', 'resolved', 'closed').
        priority: Filter by priority ('low', 'medium', 'high', 'urgent').
        limit: Max tickets to return (default 10, cap 50).

    Returns:
        Formatted ticket list, or a message if no match found.
    """
    limit = min(limit, 50)

    sql = "SELECT * FROM tickets WHERE 1=1"
    params = []
    if customer_id is not None:
        sql += " AND customer_id = ?"
        params.append(customer_id)
    if status:
        sql += " AND status = ?"
        params.append(status)
    if priority:
        sql += " AND priority = ?"
        params.append(priority)
    sql += " ORDER BY created_at DESC LIMIT ?"
    params.append(limit)

    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    rows = conn.execute(sql, params).fetchall()
    conn.close()

    if not rows:
        return "No tickets found matching those filters."

    out = []
    for r in rows:
        out.append(
            f"Ticket #{r['ticket_id']} (Customer #{r['customer_id']})\n"
            f"  Subject: {r['subject']}\n"
            f"  Status: {r['status']}  |  Priority: {r['priority']}\n"
            f"  Created: {r['created_at']}\n"
            f"  Description: {r['description']}\n"
            f"  Resolution: {r['resolution'] or '(none yet)'}"
        )
    return "\n\n".join(out)


# TOOL 3: Query a customer's orders
@mcp.tool()
def query_orders(
    customer_id: Optional[int] = None,
    status: Optional[str] = None,
    limit: int = 10,
) -> str:
    """Fetch customer orders, optionally filtered.

    Args:
        customer_id: If provided, only orders for this customer are returned.
        status: Filter by status ('pending', 'shipped', 'delivered', 'refunded').
        limit: Max orders to return (default 10, cap 50).

    Returns:
        Formatted order list, or a message if no match found.
    """
    limit = min(limit, 50)

    sql = "SELECT * FROM orders WHERE 1=1"
    params = []
    if customer_id is not None:
        sql += " AND customer_id = ?"
        params.append(customer_id)
    if status:
        sql += " AND status = ?"
        params.append(status)
    sql += " ORDER BY order_date DESC LIMIT ?"
    params.append(limit)

    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    rows = conn.execute(sql, params).fetchall()
    conn.close()

    if not rows:
        return "No orders found matching those filters."

    out = []
    for r in rows:
        out.append(
            f"Order #{r['order_id']} (Customer #{r['customer_id']})\n"
            f"  Product: {r['product']}\n"
            f"  Amount: ${r['amount_usd']:.2f}\n"
            f"  Status: {r['status']}\n"
            f"  Date: {r['order_date']}"
        )
    return "\n\n".join(out)


# TOOL 4: Run a read-only SQL query
@mcp.tool()
def run_sql(sql_query: str) -> str:
    """Execute a read-only SQL query against the customer database.

    IMPORTANT: Only SELECT statements are allowed. Any other statement
    (INSERT, UPDATE, DELETE, DROP, etc.) is rejected.

    Schema:
      customers(customer_id, name, email, plan, signup_date, country)
      tickets(ticket_id, customer_id, subject, description, status,
              priority, created_at, resolution)
      orders(order_id, customer_id, product, amount_usd, order_date, status)

    Args:
        sql_query: A single SELECT statement.

    Returns:
        Query results as formatted text, or an error message.
    """
    cleaned = sql_query.strip().rstrip(";").strip()
    if not cleaned.lower().startswith("select"):
        return "ERROR: Only SELECT queries are permitted."
    forbidden = ["insert", "update", "delete", "drop", "alter",
                 "create", "truncate", "replace", "attach"]
    lowered = cleaned.lower()
    if any(f" {kw} " in f" {lowered} " for kw in forbidden):
        return "ERROR: Query contains a forbidden keyword."

    try:
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row
        rows = conn.execute(cleaned).fetchall()
        conn.close()
    except Exception as e:
        return f"ERROR executing query: {e}"

    if not rows:
        return "(query returned no rows)"

    # Truncate large result sets for LLM-friendly output
    if len(rows) > 30:
        rows = rows[:30]
        suffix = "\n\n(truncated to first 30 rows)"
    else:
        suffix = ""

    header = " | ".join(rows[0].keys())
    body = "\n".join(" | ".join(str(v) for v in r) for r in rows)
    return f"{header}\n{body}{suffix}"


# TOOL 5: Search the policy PDFs
@mcp.tool()
def search_policy_documents(query: str, k: int = 6) -> str:
    """Search company policy PDFs for passages relevant to the question.

    Args:
        query: A natural-language question or topic.
        k: Number of passages to return (default 6, cap 10).

    Returns:
        The top-k relevant policy passages with source file citations.
    """
    k = min(max(k, 1), 10)
    docs = _vectorstore.similarity_search(query, k=k)
    if not docs:
        return "No relevant policy passages found."

    out = []
    for i, d in enumerate(docs, 1):
        src = d.metadata.get("source_file", "unknown.pdf")
        page = d.metadata.get("page", "?")
        out.append(f"[Passage {i} | {src} | page {page}]\n{d.page_content}")
    return "\n\n".join(out)


# Run the server
if __name__ == "__main__":
    # stdio transport: the server reads/writes on stdin/stdout.
    # Clients spawn this script as a subprocess and talk to it that way.
    mcp.run(transport="stdio")