"""
Step 2 - Ingestion
PDFs -> pages (with page numbers) -> chunks -> embeddings -> ChromaDB

Run:  python -m src.ingest
"""
import re
import shutil
import time

from pypdf import PdfReader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

from src.config import (
    DATA_DIR, CHROMA_DIR, EMBEDDING_MODEL,
    CHUNK_SIZE, CHUNK_OVERLAP, COLLECTION_NAME,
)


def pretty_title(filename: str) -> str:
    """'01_Work_From_Home_Policy.pdf' -> 'Work From Home Policy'"""
    name = re.sub(r"\.pdf$", "", filename, flags=re.IGNORECASE)
    name = re.sub(r"^\d+_", "", name)
    return name.replace("_", " ")


def clean_page_text(text: str) -> str:
    """Remove repeating header/footer lines and extra spaces."""
    lines = []
    for line in text.splitlines():
        low = line.lower()
        if low.startswith("nexora technologies pvt. ltd.") and "|" in line:
            continue  # footer
        if "internal use - employees of" in low:
            continue  # header
        lines.append(line.strip())
    text = "\n".join(lines)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def load_pdfs() -> list[Document]:
    """Read every PDF; one Document per page, with source + page metadata."""
    pdf_files = sorted(DATA_DIR.glob("*.pdf"))
    if not pdf_files:
        raise FileNotFoundError(f"No PDFs found in {DATA_DIR}")

    docs = []
    for pdf_path in pdf_files:
        reader = PdfReader(str(pdf_path))
        pages_added = 0
        for page_no, page in enumerate(reader.pages, start=1):
            text = clean_page_text(page.extract_text() or "")
            if not text:
                continue
            docs.append(Document(
                page_content=text,
                metadata={
                    "source": pretty_title(pdf_path.name),
                    "file": pdf_path.name,
                    "page": page_no,
                },
            ))
            pages_added += 1
        print(f"  loaded {pdf_path.name}: {pages_added} page(s)")
    return docs


def chunk_documents(docs: list[Document]) -> list[Document]:
    """Split pages into overlapping chunks. Metadata is copied to each chunk."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    return splitter.split_documents(docs)


def get_embeddings() -> HuggingFaceEmbeddings:
    """Free local embedding model (downloads ~90 MB the first time)."""
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        encode_kwargs={"normalize_embeddings": True},
    )


def build_vector_store() -> Chroma:
    """Full pipeline: load -> chunk -> embed -> save to ChromaDB."""
    print("\n[1/4] Reading PDFs...")
    docs = load_pdfs()
    print(f"      total pages: {len(docs)}")

    print("[2/4] Chunking...")
    chunks = chunk_documents(docs)
    print(f"      total chunks: {len(chunks)}")

    print("[3/4] Loading embedding model (first run downloads it)...")
    embeddings = get_embeddings()

    print("[4/4] Building ChromaDB (old index is replaced)...")
    if CHROMA_DIR.exists():
        shutil.rmtree(CHROMA_DIR, ignore_errors=True)

    store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name=COLLECTION_NAME,
        persist_directory=str(CHROMA_DIR),
    )
    print(f"      saved to: {CHROMA_DIR}")
    return store


def get_vector_store() -> Chroma:
    """Open the existing index; build it first if it does not exist yet."""
    if not CHROMA_DIR.exists() or not any(CHROMA_DIR.iterdir()):
        return build_vector_store()
    return Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=get_embeddings(),
        persist_directory=str(CHROMA_DIR),
    )


if __name__ == "__main__":
    start = time.time()
    store = build_vector_store()
    print(f"\nDone in {time.time() - start:.1f} seconds")

    # ---- quick test search ----
    test_queries = [
        "How many WFH days can I take per month?",
        "What is the hotel limit in metro cities?",
        "How long is maternity leave?",
    ]
    for q in test_queries:
        print("\n" + "=" * 60)
        print("Q:", q)
        for doc, score in store.similarity_search_with_score(q, k=2):
            m = doc.metadata
            snippet = doc.page_content[:150].replace("\n", " ")
            print(f"  - {m['source']} | page {m['page']} | distance {score:.3f}")
            print(f"    {snippet}...")