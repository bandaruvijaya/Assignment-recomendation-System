import pandas as pd

INPUT = "data/shl_catalog.csv"
OUTPUT = "data/shl_catalog_clean.csv"

def main():
    df = pd.read_csv(INPUT)

    # Drop invalid rows
    df = df.dropna(subset=["assessment_name", "url"])
    df = df.drop_duplicates(subset=["assessment_name"])

    # Normalize text fields
    df["assessment_name"] = df["assessment_name"].astype(str).str.strip()
    df["description"] = df.get("description", "").fillna("").astype(str)
    df["category"] = df.get("category", "General").fillna("General").astype(str)
    df["test_type"] = df.get("test_type", "K").fillna("K").astype(str)

    # Build full_text for embeddings
    df["full_text"] = (
        df["assessment_name"] + " " +
        df["description"] + " " +
        df["category"] + " " +
        df["test_type"]
    )

    df.to_csv(OUTPUT, index=False)
    print("✅ Saved:", OUTPUT)
    print("✅ Records:", len(df))

if __name__ == "__main__":
    main()
