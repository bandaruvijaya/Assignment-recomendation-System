import pandas as pd
import requests
from urllib.parse import urlparse

API_URL = "http://127.0.0.1:8000/recommend"
DATA_FILE = "Gen_AI Dataset.xlsx"

def get_slug(url):
    """
    Normalize SHL URLs by extracting the last path component
    """
    if not isinstance(url, str):
        return ""
    path = urlparse(url).path.strip("/")
    return path.split("/")[-1]

def recall_at_10(pred_urls, true_url):
    true_slug = get_slug(true_url)
    pred_slugs = [get_slug(u) for u in pred_urls[:10]]
    return 1 if true_slug in pred_slugs else 0

def main():
    df = pd.read_excel(DATA_FILE, sheet_name="Train-Set")

    recalls = []

    for i, row in df.iterrows():
        query = row["Query"]
        true_url = row["Assessment_url"]

        resp = requests.post(API_URL, json={"query": query})
        resp.raise_for_status()

        recs = resp.json()["recommendations"]
        pred_urls = [r["assessment_url"] for r in recs]

        r10 = recall_at_10(pred_urls, true_url)
        recalls.append(r10)

        print(f"[{i+1}/{len(df)}] Recall@10 = {r10}")

    mean_recall = sum(recalls) / len(recalls)
    print("\n✅ Mean Recall@10:", round(mean_recall, 3))

if __name__ == "__main__":
    main()