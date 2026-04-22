"""
ingest.py
Reads all PDFs from data/policies/, chunks them, creates embeddings,
and stores them in a local ChromaDB vector database.

Run once after adding/updating PDFs:  python src/ingest.py
"""

import shutil
from pathlib import Path

from langchain_community.document_loaders import PDFPlumberLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

# Paths
PROJECT_ROOT = Path(__file__).parent.parent
PDF_DIR      = PROJECT_ROOT / "data" / "policies"
VECTOR_DIR   = PROJECT_ROOT / "data" / "chroma_db"

# 1. Wipe the old vector store so we start fresh every run
if VECTOR_DIR.exists():
    shutil.rmtree(VECTOR_DIR)
VECTOR_DIR.mkdir(parents=True, exist_ok=True)

# 2. Load every PDF and split into chunks
pdf_files = list(PDF_DIR.glob("*.pdf"))
if not pdf_files:
    raise SystemExit(f"No PDFs found in {PDF_DIR}. Add some and re-run.")

print(f"Found {len(pdf_files)} PDF(s):")
for p in pdf_files:
    print(f"  - {p.name}")

# chunk_size=500 chars, overlap=50 chars. Overlap preserves context across chunk boundaries (a sentence won't get cut).
splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=75,
    separators=["\n\n", "\n", ". ", " ", ""],
)

all_chunks = []
for pdf_path in pdf_files:
    loader = PDFPlumberLoader(str(pdf_path))
    pages  = loader.load()                        # one Document per page
    chunks = splitter.split_documents(pages)       # split each page further

    # Tag every chunk with its source filename (used later for citations)
    for c in chunks:
        c.metadata["source_file"] = pdf_path.name

    all_chunks.extend(chunks)
    print(f"  {pdf_path.name}: {len(pages)} pages → {len(chunks)} chunks")

print(f"\nTotal chunks: {len(all_chunks)}")

# 3. Create embeddings + store in Chroma
#    Model: all-MiniLM-L6-v2 — small, fast, runs locally, free.
#    First run will download ~90 MB of model weights.
print("\nLoading embedding model (first run downloads ~90 MB)...")
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True},
)

print("Embedding chunks and writing to Chroma...")
vectorstore = Chroma.from_documents(
    documents=all_chunks,
    embedding=embeddings,
    persist_directory=str(VECTOR_DIR),
)

print(f"\nVector store created at {VECTOR_DIR}")
print(f"   {len(all_chunks)} chunks indexed across {len(pdf_files)} PDF(s)")

# 4. Sanity check — run a test query
print("\n--- Sanity check ---")
test_query = "How long do I have to return an item?"
results = vectorstore.similarity_search(test_query, k=6)
print(f"Query: {test_query}\n")
for i, r in enumerate(results, 1):
    print(f"Result {i} (from {r.metadata.get('source_file')}):")
    print(r.page_content[:200].replace("\n", " "))
    print("-" * 60)