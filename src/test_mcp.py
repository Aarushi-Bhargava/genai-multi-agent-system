"""
test_mcp.py
Quick sanity test: spin up the MCP server, connect as a client,
list the tools, and call each one.

Run:  python src/test_mcp.py
"""

import asyncio
from pathlib import Path
from langchain_mcp_adapters.client import MultiServerMCPClient

SERVER_PATH = Path(__file__).parent / "mcp_server.py"

async def main():
    client = MultiServerMCPClient({
        "support": {
            "command": "python",
            "args": [str(SERVER_PATH)],
            "transport": "stdio",
        }
    })

    tools = await client.get_tools()
    print(f"Connected. Server exposes {len(tools)} tools:\n")
    for t in tools:
        print(f"  • {t.name}")
        print(f"      {t.description.splitlines()[0]}")
    print()

    # Call query_customer_profile with an empty string to list first 5 customers
    tool_by_name = {t.name: t for t in tools}

    print("--- calling query_customer_profile('a') ---")
    result = await tool_by_name["query_customer_profile"].ainvoke({"name_or_email": "a"})
    print(result[:500])
    print()

    print("--- calling query_tickets(status='urgent', limit=3) ---")
    result = await tool_by_name["query_tickets"].ainvoke({"status": "open", "limit": 3})
    print(result[:500])
    print()

    print("--- calling search_policy_documents('return window') ---")
    result = await tool_by_name["search_policy_documents"].ainvoke({"query": "return window"})
    print(result[:500])
    print()

    print("--- calling run_sql('SELECT COUNT(*) AS total FROM customers') ---")
    result = await tool_by_name["run_sql"].ainvoke({"sql_query": "SELECT COUNT(*) AS total FROM customers"})
    print(result)
    print()

    print("All tests completed.")

if __name__ == "__main__":
    asyncio.run(main())