import pandas as pd
import requests

API_URL = "http://127.0.0.1:8000/recommend"
DATA_FILE = "Gen_AI Dataset.xlsx"
OUTPUT_FILE = "outputs/predictions.csv"

def main():
    df = pd.read_excel(DATA_FILE, sheet_name="Test-Set")

    rows = []

    for _, row in df.iterrows():
        query = row["Query"]

        resp = requests.post(API_URL, json={"query": query})
        resp.raise_for_status()

        recs = resp.json()["recommendations"]

        for r in recs:
            rows.append({
                "Query": query,
                "Assessment_url": r["assessment_url"]
            })

    out_df = pd.DataFrame(rows)
    out_df.to_csv(OUTPUT_FILE, index=False)

    print("✅ predictions.csv generated")
    print("✅ Rows:", len(out_df))

if __name__ == "__main__":
    main()
