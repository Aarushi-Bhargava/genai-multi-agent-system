"""
rag_agent.py
A standalone RAG agent that answers natural-language questions
using the policy PDFs indexed in ChromaDB.

Run directly for testing: python src/rag_agent.py
"""

from pathlib import Path
from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# 1. Load environment (Groq API key)
load_dotenv()

# 2. Connect to the existing Chroma vector store
PROJECT_ROOT = Path(__file__).parent.parent
VECTOR_DIR   = PROJECT_ROOT / "data" / "chroma_db"

if not VECTOR_DIR.exists():
    raise SystemExit(
        f"No vector store found at {VECTOR_DIR}. "
        "Run 'python src/ingest.py' first."
    )

# Must use the SAME embedding model used during ingestion, otherwise query embeddings won't be comparable to stored ones.
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True},
)

vectorstore = Chroma(
    persist_directory=str(VECTOR_DIR),
    embedding_function=embeddings,
)

# A "retriever" is a thin wrapper around the vector store that returns the top-k most similar chunks for a given query.
retriever = vectorstore.as_retriever(search_kwargs={"k": 10})

# 3. Initialize the LLM (same Groq model as the SQL agent)
llm = ChatGroq(
    model="openai/gpt-oss-120b",
    temperature=0,
)

# 4. The RAG prompt — engineered for grounding + citations
RAG_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "You are a customer support policy assistant. Your job is to answer "
     "questions about company policies using ONLY the provided context from "
     "official policy documents.\n\n"
     "Strict rules:\n"
     "1. Answer ONLY from the provided context below. Do not use outside knowledge.\n"
     "2. If the context does not contain the answer, say exactly: "
     "'I couldn't find that information in the available policy documents.'\n"
     "3. Always cite the source file(s) you used, formatted as "
     "'(Source: filename.pdf)'.\n"
     "4. Quote exact policy language when it's important (e.g., timeframes, fees).\n"
     "5. If multiple policies apply, mention each and cite each source.\n"
     "6. Keep answers concise and factual. No speculation.\n\n"
     "Context:\n{context}"
    ),
    ("user", "{question}"),
])

# 5. Helper to format retrieved chunks into a labeled context string
def format_context(docs) -> str:
    """Turn a list of Chroma Documents into a labeled string with source tags."""
    if not docs:
        return "(no relevant policy content retrieved)"
    chunks = []
    for i, d in enumerate(docs, 1):
        source = d.metadata.get("source_file", "unknown.pdf")
        page   = d.metadata.get("page", "?")
        chunks.append(f"[Chunk {i} | {source} | page {page}]\n{d.page_content}")
    return "\n\n".join(chunks)

# 6. The RAG pipeline: retrieve -> format -> prompt -> LLM -> string
def ask(question: str) -> str:
    """Answer a policy question using RAG over the indexed PDFs."""
    try:
        # Retrieval step
        docs = retriever.invoke(question)
        context = format_context(docs)

        # Generation step
        chain = RAG_PROMPT | llm | StrOutputParser()
        answer = chain.invoke({"context": context, "question": question})
        return answer
    except Exception as e:
        return f"Sorry, I hit an error: {e}"

# 7. Test
if __name__ == "__main__":
    test_questions = [
        "How many days do I have to return an item?",
        "What is the AppleCare+ service fee for iPhone screen damage?",
        "Can I return an Apple Gift Card?",
        "What happens if I cancel AppleCare+ within 30 days?",
        "What's the warranty on a refrigerator?",   #out of scope on purpose
    ]

    for q in test_questions:
        print("\n" + "=" * 70)
        print(f"Q: {q}")
        print("-" * 70)
        print(ask(q))