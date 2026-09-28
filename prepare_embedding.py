"""modify:rag-embedding. Download once; normal retrieval is local and never falls back."""
from pathlib import Path
from sentence_transformers import SentenceTransformer

def main():
    root=Path(__file__).resolve().parent
    model=SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2',device='cpu', revision='1110a243fdf4706b3f48f1d95db1a4f5529b4d41')
    model.save(str(root/'agent/models/embedding'))
if __name__=='__main__': main()
