import pandas as pd
import pickle
import faiss
from sentence_transformers import SentenceTransformer
import numpy as np

INPUT = "data/shl_catalog_clean.csv"
INDEX_OUT = "embeddings/faiss.index"
META_OUT = "embeddings/metadata.pkl"

def main():
    print("📥 Loading data...")
    df = pd.read_csv(INPUT)

    texts = df["full_text"].tolist()
    metadata = df[["assessment_name", "url", "test_type", "category"]].to_dict(orient="records")

    print("🤖 Loading embedding model...")
    model = SentenceTransformer("all-MiniLM-L6-v2")

    print("🔢 Generating embeddings...")
    embeddings = model.encode(
        texts,
        batch_size=32,
        show_progress_bar=True,
        normalize_embeddings=True
    )

    embeddings = np.array(embeddings).astype("float32")

    print("🧠 Building FAISS index...")
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)  # cosine similarity via normalized vectors
    index.add(embeddings)

    print("💾 Saving index and metadata...")
    faiss.write_index(index, INDEX_OUT)
    with open(META_OUT, "wb") as f:
        pickle.dump(metadata, f)

    print("✅ FAISS index saved:", INDEX_OUT)
    print("✅ Metadata saved:", META_OUT)
    print("✅ Total vectors:", index.ntotal)

if __name__ == "__main__":
    main()
