import os
import requests
import feedparser
from datetime import datetime
import json
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

# -----------------------------
# CONFIG
# -----------------------------
topic_query = "uncertainty prediction ML"
START_DATE = datetime(2025, 9, 1)
END_DATE = datetime(2025, 9, 30, 23, 59, 59)
OUTPUT_DIR = "arxiv_papers_sept2025"
os.makedirs(OUTPUT_DIR, exist_ok=True)
base_url = "http://export.arxiv.org/api/query"

N_CLUSTERS = 5
# -----------------------------


# -----------------------------
# STEP 1: Fetch papers from arXiv
# -----------------------------
def fetch_papers(query, max_results=200):
    params = {
        "search_query": query,
        "start": 0,
        "max_results": max_results,
        "sortBy": "submittedDate",
        "sortOrder": "descending",
    }
    response = requests.get(base_url, params=params)
    response.raise_for_status()
    feed = feedparser.parse(response.text)
    return feed.entries


def is_september_2025(date_str):
    date = datetime.strptime(date_str, "%Y-%m-%dT%H:%M:%SZ")
    return START_DATE <= date <= END_DATE


def download_file(url, filepath):
    r = requests.get(url, stream=True)
    r.raise_for_status()
    with open(filepath, "wb") as f:
        for chunk in r.iter_content(chunk_size=8192):
            f.write(chunk)


print("Fetching recent papers from arXiv...")
papers = fetch_papers(topic_query, max_results=200)

for entry in papers:
    if not is_september_2025(entry.published):
        continue

    title = entry.title.replace("\n", " ").strip()
    abstract = entry.summary.replace("\n", " ").strip()
    published = entry.published
    pdf_url = None
    doi = entry.get("arxiv_doi", entry.get("id"))

    for link in entry.links:
        if link.rel == "related" and "doi.org" in link.href:
            doi = link.href
        if link.get("title") == "pdf":
            pdf_url = link.href

    if not pdf_url:
        continue

    arxiv_id = entry.id.split("/")[-1]
    pdf_filename = os.path.join(OUTPUT_DIR, f"{arxiv_id}.pdf")
    txt_filename = os.path.join(OUTPUT_DIR, f"{arxiv_id}.txt")

    if not os.path.exists(pdf_filename):
        print(f"Downloading PDF: {title}")
        download_file(pdf_url, pdf_filename)

    with open(txt_filename, "w", encoding="utf-8") as f:
        f.write(f"Title: {title}\n")
        f.write(f"Published: {published}\n")
        f.write(f"DOI/ID: {doi}\n")
        f.write("Abstract:\n")
        f.write(abstract + "\n")

print("\n✅ Download completed. Proceeding to embeddings & clustering...\n")

# -----------------------------
# STEP 2: Generate embeddings using SentenceTransformer (LOCAL)
# Because Open AI rate limit got exceeded and causing script to crash.
# -----------------------------
print("Loading sentence-transformers model...")
model = SentenceTransformer("all-MiniLM-L6-v2")

texts = []
file_names = []

for file in os.listdir(OUTPUT_DIR):
    if file.endswith(".txt"):
        with open(os.path.join(OUTPUT_DIR, file), "r", encoding="utf-8") as f:
            content = f.read()
            if "Abstract:" in content:
                abstract = content.split("Abstract:")[1].strip()
            else:
                abstract = content
            texts.append(abstract)
            file_names.append(file)

print(f"Loaded {len(texts)} abstracts for embedding generation.\n")

print("Generating embeddings locally (this may take a minute)...")
embeddings = model.encode(texts, show_progress_bar=True, convert_to_numpy=True)

np.save(os.path.join(OUTPUT_DIR, "embeddings.npy"), embeddings)

with open(os.path.join(OUTPUT_DIR, "meta.json"), "w", encoding="utf-8") as f:
    json.dump(
        [{"filename": fn, "text": txt} for fn, txt in zip(file_names, texts)],
        f, ensure_ascii=False, indent=2
    )

print(f"✅ Saved embeddings and metadata inside {OUTPUT_DIR}\n")

# -----------------------------
# STEP 3: Clustering
# -----------------------------
print("Running KMeans clustering...")
kmeans = KMeans(n_clusters=N_CLUSTERS, random_state=42, n_init=10)
labels = kmeans.fit_predict(embeddings)

with open(os.path.join(OUTPUT_DIR, "cluster_assignments.csv"), "w", encoding="utf-8") as f:
    f.write("filename,cluster\n")
    for fn, lbl in zip(file_names, labels):
        f.write(f"{fn},{lbl}\n")

print(f"✅ Clustering completed and saved to cluster_assignments.csv\n")

# -----------------------------
# STEP 4: Find representative paper per cluster
# -----------------------------
print("Finding representative abstracts...")
representatives = []
for i in range(N_CLUSTERS):
    cluster_indices = np.where(labels == i)[0]
    cluster_vectors = embeddings[cluster_indices]
    centroid = kmeans.cluster_centers_[i]
    sims = cosine_similarity([centroid], cluster_vectors)[0]
    best_idx = cluster_indices[np.argmax(sims)]
    representatives.append((i, file_names[best_idx], texts[best_idx]))

with open(os.path.join(OUTPUT_DIR, "representatives.txt"), "w", encoding="utf-8") as f:
    for cid, fname, abstract in representatives:
        f.write("=" * 80 + "\n")
        f.write(f"Cluster {cid} Representative File: {fname}\n")
        f.write(f"Abstract:\n{abstract}\n\n")

print("✅ Representative abstracts saved to representatives.txt")
print("\n🎯 All steps (Fetch → Embeddings → Clustering → Representatives) completed successfully!")
# Output is in arxiv_papers_sept2025 folders with 4 files
# 1) cluster_assignments.csv
# embeddings.npy
# meta.json
# representatives.txt

# The below run will help analysing computed output.
emb = np.load("arxiv_papers_sept2025/embeddings.npy")
print(emb.shape)
