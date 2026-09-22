from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = BASE_DIR / "outputs"

KB_FILE = DATA_DIR / "industrial_kb.json"
VECTOR_INDEX_FILE = OUTPUT_DIR / "vector_index.npz"
CHUNK_FILE = OUTPUT_DIR / "chunks.json"
RESULT_JSON_FILE = OUTPUT_DIR / "retrieval_results.json"
RESULT_CSV_FILE = OUTPUT_DIR / "retrieval_results.csv"

CHUNK_SIZE = 50
CHUNK_OVERLAP = 10
TOP_K = 3
SIMILARITY_THRESHOLD = 0.35

DATA_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
