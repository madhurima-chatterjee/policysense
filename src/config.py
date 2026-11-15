"""
Central settings for PolicySense.
Every other file imports from here, so change settings in one place only.
"""
import os
from pathlib import Path
from dotenv import load_dotenv

# ---------- Paths ----------
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data" / "policies"
EVAL_FILE = BASE_DIR / "data" / "eval" / "test_questions.csv"
CHROMA_DIR = BASE_DIR / "chroma_db"

# ---------- Environment variables ----------
load_dotenv(BASE_DIR / ".env")
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")

# ---------- Embeddings ----------
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# ---------- Chunking ----------
CHUNK_SIZE = 800
CHUNK_OVERLAP = 100

# ---------- Retrieval ----------
COLLECTION_NAME = "policysense"
TOP_K = 6                 # number of chunks sent to the LLM
MAX_HISTORY_TURNS = 4     # how many past Q&A pairs the bot remembers

# ---------- Messages ----------
NO_ANSWER_MESSAGE = (
    "Sorry, I could not find an answer to this question in the available "
    "policy documents. Please contact HR at hr.helpdesk@nexora.example."
)


def check_setup() -> bool:
    """Print a checklist so you can see exactly what is missing."""
    ok = True
    print("=" * 55)
    print("PolicySense - setup check")
    print("=" * 55)

    print(f"Project folder : {BASE_DIR}")

    # 1. API key
    if GROQ_API_KEY and GROQ_API_KEY.startswith("gsk_"):
        print(f"[OK]   GROQ_API_KEY found ({GROQ_API_KEY[:8]}...)")
    else:
        print("[FAIL] GROQ_API_KEY missing or invalid. Check your .env file.")
        ok = False

    print(f"[INFO] Groq model: {GROQ_MODEL}")

    # 2. PDFs
    pdfs = sorted(DATA_DIR.glob("*.pdf")) if DATA_DIR.exists() else []
    if pdfs:
        print(f"[OK]   {len(pdfs)} PDF(s) found in data/policies:")
        for p in pdfs:
            print(f"         - {p.name}")
    else:
        print("[FAIL] No PDFs found in data/policies. Copy the 8 PDFs there.")
        ok = False

    # 3. Test questions
    if EVAL_FILE.exists():
        print("[OK]   test_questions.csv found")
    else:
        print("[WARN] data/eval/test_questions.csv not found (needed only in Step 5)")

    print("=" * 55)
    print("RESULT:", "ALL GOOD, ready for Step 2" if ok else "FIX THE FAIL ITEMS ABOVE")
    return ok


if __name__ == "__main__":
    check_setup()