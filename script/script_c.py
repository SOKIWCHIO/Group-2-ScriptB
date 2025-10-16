import os
import requests
import feedparser
from datetime import datetime, timedelta
import json
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import time

# -----------------------------
# CONFIG
# -----------------------------
# Expanded SPECIFIC search terms for ML uncertainty papers
search_terms = [
    # Original specific terms
    'ti:"uncertainty quantification" AND (cat:cs.LG OR cat:cs.AI OR cat:stat.ML)',
    'ti:"predictive uncertainty" AND (cat:cs.LG OR cat:cs.AI OR cat:stat.ML)',
    'ti:"epistemic uncertainty" AND (cat:cs.LG OR cat:cs.AI OR cat:stat.ML)',
    'ti:"aleatoric uncertainty" AND (cat:cs.LG OR cat:cs.AI OR cat:stat.ML)',
    'abs:"uncertainty estimation" AND (cat:cs.LG OR cat:cs.AI OR cat:stat.ML)',
    'abs:"confidence calibration" AND (cat:cs.LG OR cat:cs.AI OR cat:stat.ML)',
    'abs:"bayesian deep learning" AND (cat:cs.LG OR cat:cs.AI OR cat:stat.ML)',
    'abs:"probabilistic prediction" AND (cat:cs.LG OR cat:cs.AI OR cat:stat.ML)',
    'abs:"uncertainty aware" AND (cat:cs.LG OR cat:cs.AI OR cat:stat.ML)',
    'abs:"monte carlo dropout" AND (cat:cs.LG OR cat:cs.AI OR cat:stat.ML)',
    'abs:"ensemble uncertainty" AND (cat:cs.LG OR cat:cs.AI OR cat:stat.ML)',
    'abs:"prediction intervals" AND (cat:cs.LG OR cat:cs.AI OR cat:stat.ML)',
    'abs:"conformal prediction" AND (cat:cs.LG OR cat:cs.AI OR cat:stat.ML)',
    
    # Additional SPECIFIC uncertainty-related terms
    'ti:"uncertainty propagation" AND (cat:cs.LG OR cat:cs.AI OR cat:stat.ML)',
    'ti:"prediction uncertainty" AND (cat:cs.LG OR cat:cs.AI OR cat:stat.ML)',
    'ti:"model uncertainty" AND (cat:cs.LG OR cat:cs.AI OR cat:stat.ML)',
    'abs:"deep ensembles" AND (cat:cs.LG OR cat:cs.AI)',
    'abs:"evidential deep learning" AND (cat:cs.LG OR cat:cs.AI)',
    'abs:"prior networks" AND (cat:cs.LG OR cat:cs.AI)',
    'abs:"posterior networks" AND (cat:cs.LG OR cat:cs.AI)',
    'abs:"normalizing flows" AND abs:"uncertainty" AND cat:cs.LG',
    'abs:"temperature scaling" AND (cat:cs.LG OR cat:cs.AI)',
    'abs:"expected calibration error" AND cat:cs.LG',
    'abs:"maximum mean calibration error" AND cat:cs.LG',
    'abs:"reliable confidence" AND (cat:cs.LG OR cat:cs.AI)',
    'abs:"selective prediction" AND cat:cs.LG',
    'abs:"predictive entropy" AND (cat:cs.LG OR cat:cs.AI)',
    'abs:"mutual information" AND abs:"uncertainty" AND cat:cs.LG',
    'abs:"BALD" AND (cat:cs.LG OR cat:cs.AI)',  # Bayesian Active Learning by Disagreement
    'abs:"active learning" AND abs:"uncertainty" AND cat:cs.LG',
    'abs:"OOD detection" AND (cat:cs.LG OR cat:cs.AI)',
    'abs:"out-of-distribution" AND abs:"uncertainty" AND cat:cs.LG',
    'abs:"distributional shift" AND (cat:cs.LG OR cat:stat.ML)',
    'abs:"covariate shift" AND abs:"uncertainty" AND cat:cs.LG',
    'abs:"label shift" AND abs:"uncertainty" AND cat:cs.LG',
    'abs:"domain shift" AND abs:"prediction" AND cat:cs.LG',
    'abs:"test-time adaptation" AND cat:cs.LG',
    'abs:"prediction sets" AND (cat:cs.LG OR cat:stat.ML)',
    'abs:"coverage guarantee" AND (cat:cs.LG OR cat:stat.ML)',
    'abs:"conditional coverage" AND cat:stat.ML',
    'abs:"distribution-free uncertainty" AND cat:cs.LG',
    'abs:"probabilistic forecasting" AND cat:cs.LG',
    'abs:"interval prediction" AND cat:cs.LG',
    'abs:"quantile forecasting" AND (cat:cs.LG OR cat:stat.ML)',
    
    # More specific technical terms
    'abs:"variational inference" AND abs:"uncertainty" AND cat:cs.LG',
    'abs:"gaussian processes" AND abs:"prediction" AND cat:cs.LG',
    'abs:"posterior sampling" AND (cat:cs.LG OR cat:stat.ML)',
    'abs:"credible intervals" AND (cat:cs.LG OR cat:stat.ML)',
    'abs:"prediction consistency" AND cat:cs.LG',
    'abs:"model confidence" AND cat:cs.LG',
    'abs:"uncertainty decomposition" AND cat:cs.LG',
    'abs:"heteroscedastic uncertainty" AND cat:cs.LG',
    'abs:"homoscedastic uncertainty" AND cat:cs.LG',
    'abs:"uncertainty budget" AND cat:cs.LG',
    'abs:"reliability diagram" AND cat:cs.LG',
    'abs:"calibration error" AND cat:cs.LG',
    'abs:"Platt scaling" AND cat:cs.LG',
    'abs:"isotonic regression" AND abs:"calibration" AND cat:cs.LG',
    'abs:"histogram binning" AND abs:"calibration" AND cat:cs.LG',
]

# Extended date range to get more papers (3 years)
START_DATE = datetime(2022, 1, 1)  # Go back to Jan 2022
END_DATE = datetime(2025, 10, 15)  # Current date

OUTPUT_DIR = "arxiv_ml_uncertainty_papers_1500"
os.makedirs(OUTPUT_DIR, exist_ok=True)
base_url = "http://export.arxiv.org/api/query"

N_CLUSTERS = 15  # Increased clusters for 1500 papers
TARGET_PAPERS = 1500
# -----------------------------


# -----------------------------
# STEP 1: Fetch papers from arXiv with better filtering
# -----------------------------
def fetch_papers_batch(query, start=0, max_results=100):
    """Fetch a batch of papers from arXiv API"""
    params = {
        "search_query": query,
        "start": start,
        "max_results": max_results,
        "sortBy": "submittedDate",
        "sortOrder": "descending",
    }
    
    try:
        response = requests.get(base_url, params=params, timeout=30)
        response.raise_for_status()
        feed = feedparser.parse(response.text)
        return feed.entries
    except Exception as e:
        print(f"Error fetching batch: {e}")
        return []


def fetch_all_papers_for_query(query, max_total=300):  # Increased from 200 to 300
    """Fetch all papers for a given query with pagination"""
    all_papers = []
    batch_size = 100
    start = 0
    
    while len(all_papers) < max_total:
        print(f"  Fetching batch starting at {start}...")
        batch = fetch_papers_batch(query, start, batch_size)
        
        if not batch:
            break
            
        all_papers.extend(batch)
        
        if len(batch) < batch_size:
            break
            
        start += batch_size
        time.sleep(3)  # Be respectful to arXiv API
        
    return all_papers[:max_total]


def is_in_date_range(date_str):
    """Check if paper is within our date range"""
    try:
        date = datetime.strptime(date_str, "%Y-%m-%dT%H:%M:%SZ")
        return START_DATE <= date <= END_DATE
    except:
        return False


def is_ml_related(entry):
    """Filter to ensure paper is ML/CS related based on categories"""
    categories = entry.get('tags', [])
    if not categories:
        categories = entry.get('arxiv_primary_category', {}).get('term', '')
        if categories:
            categories = [{'term': categories}]
    
    ml_categories = ['cs.LG', 'cs.AI', 'stat.ML', 'cs.CV', 'cs.NE', 'cs.CL']
    
    for cat in categories:
        if any(ml_cat in cat.get('term', '') for ml_cat in ml_categories):
            return True
    return False


def contains_uncertainty_keywords(title, abstract):
    """Check if paper actually relates to uncertainty/prediction"""
    text = (title + " " + abstract).lower()
    keywords = [
        'uncertainty', 'confidence', 'calibration', 'bayesian',
        'probabilistic', 'ensemble', 'prediction interval',
        'epistemic', 'aleatoric', 'monte carlo dropout',
        'variational inference', 'gaussian process', 'conformal',
        'credible interval', 'quantile regression', 'risk assessment',
        'deep ensemble', 'evidential', 'temperature scaling',
        'expected calibration error', 'reliability', 'coverage',
        'out-of-distribution', 'ood', 'distribution shift',
        'selective prediction', 'predictive entropy', 'bald',
        'active learning', 'test-time', 'prediction set',
        'platt scaling', 'isotonic regression', 'histogram binning'
    ]
    return any(keyword in text for keyword in keywords)


def download_file(url, filepath):
    """Download file from URL"""
    try:
        r = requests.get(url, stream=True, timeout=30)
        r.raise_for_status()
        with open(filepath, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
        return True
    except Exception as e:
        print(f"    Error downloading: {e}")
        return False


print(f"Fetching ML uncertainty papers from {START_DATE.date()} to {END_DATE.date()}...")
print(f"Target: {TARGET_PAPERS} papers")
print(f"Number of search queries: {len(search_terms)}\n")

# Collect all papers from different queries
all_papers_dict = {}  # Use dict to avoid duplicates

# Search through ALL terms without early stopping
for i, query in enumerate(search_terms, 1):
    print(f"Query {i}/{len(search_terms)}: {query[:70]}...")
    papers = fetch_all_papers_for_query(query, max_total=300)
    
    for entry in papers:
        arxiv_id = entry.id.split("/")[-1]
        if arxiv_id not in all_papers_dict:
            all_papers_dict[arxiv_id] = entry
    
    print(f"  Found {len(papers)} papers in this query")
    print(f"  Total unique papers collected: {len(all_papers_dict)}")
    
    # Don't break early - search all terms to maximize paper collection

print(f"\n{'='*60}")
print(f"Total unique papers found before filtering: {len(all_papers_dict)}")
print(f"{'='*60}\n")

# Filter and download papers
downloaded_count = 0
papers_metadata = []
filtered_count = 0
date_filtered = 0
category_filtered = 0
keyword_filtered = 0

print("Starting filtering and download process...")

for arxiv_id, entry in all_papers_dict.items():
    if downloaded_count >= TARGET_PAPERS:
        print(f"\nReached target of {TARGET_PAPERS} papers. Stopping downloads.")
        break
    
    # Apply filters and count what gets filtered
    if not is_in_date_range(entry.published):
        date_filtered += 1
        continue
    
    if not is_ml_related(entry):
        category_filtered += 1
        continue
    
    title = entry.title.replace("\n", " ").strip()
    abstract = entry.summary.replace("\n", " ").strip()
    
    if not contains_uncertainty_keywords(title, abstract):
        keyword_filtered += 1
        continue
    
    filtered_count += 1
    published = entry.published
    pdf_url = None
    doi = entry.get("arxiv_doi", entry.get("id"))
    
    # Get categories
    categories = []
    if 'tags' in entry:
        categories = [tag['term'] for tag in entry.tags]
    
    for link in entry.links:
        if link.rel == "related" and "doi.org" in link.href:
            doi = link.href
        if link.get("title") == "pdf":
            pdf_url = link.href
    
    if not pdf_url:
        continue
    
    pdf_filename = os.path.join(OUTPUT_DIR, f"{arxiv_id}.pdf")
    txt_filename = os.path.join(OUTPUT_DIR, f"{arxiv_id}.txt")
    
    # Download PDF if not exists
    if not os.path.exists(pdf_filename):
        print(f"Downloading [{downloaded_count + 1}/{TARGET_PAPERS}]: {title[:60]}...")
        if not download_file(pdf_url, pdf_filename):
            continue
        time.sleep(2)  # Rate limiting
    
    # Save metadata
    with open(txt_filename, "w", encoding="utf-8") as f:
        f.write(f"Title: {title}\n")
        f.write(f"Published: {published}\n")
        f.write(f"DOI/ID: {doi}\n")
        f.write(f"Categories: {', '.join(categories)}\n")
        f.write("Abstract:\n")
        f.write(abstract + "\n")
    
    papers_metadata.append({
        "arxiv_id": arxiv_id,
        "title": title,
        "abstract": abstract,
        "published": published,
        "categories": categories
    })
    
    downloaded_count += 1
    
    if downloaded_count % 50 == 0:
        print(f"  Progress: {downloaded_count}/{TARGET_PAPERS} papers downloaded")
    
    if downloaded_count % 100 == 0:
        # Save intermediate metadata in case of interruption
        with open(os.path.join(OUTPUT_DIR, "papers_metadata_temp.json"), "w", encoding="utf-8") as f:
            json.dump(papers_metadata, f, ensure_ascii=False, indent=2)

print(f"\n{'='*60}")
print("FILTERING SUMMARY")
print(f"{'='*60}")
print(f"Total papers found: {len(all_papers_dict)}")
print(f"Filtered by date range: {date_filtered}")
print(f"Filtered by category (not ML): {category_filtered}")
print(f"Filtered by keywords: {keyword_filtered}")
print(f"Papers passing all filters: {filtered_count}")
print(f"Successfully downloaded: {downloaded_count}")
print(f"{'='*60}\n")

# Save final metadata
with open(os.path.join(OUTPUT_DIR, "papers_metadata.json"), "w", encoding="utf-8") as f:
    json.dump(papers_metadata, f, ensure_ascii=False, indent=2)

# Remove temp file if exists
temp_file = os.path.join(OUTPUT_DIR, "papers_metadata_temp.json")
if os.path.exists(temp_file):
    os.remove(temp_file)

if downloaded_count == 0:
    print("⚠️ No papers were downloaded. Please check your filters and try again.")
    exit()

print("Proceeding to embeddings & clustering...\n")

# -----------------------------
# STEP 2: Generate embeddings using SentenceTransformer
# -----------------------------
print("Loading sentence-transformers model...")
model = SentenceTransformer("all-MiniLM-L6-v2")

texts = []
file_names = []
titles = []

for file in os.listdir(OUTPUT_DIR):
    if file.endswith(".txt"):
        with open(os.path.join(OUTPUT_DIR, file), "r", encoding="utf-8") as f:
            content = f.read()
            
            # Extract title and abstract
            title = ""
            if "Title:" in content:
                title = content.split("Title:")[1].split("\n")[0].strip()
            
            if "Abstract:" in content:
                abstract = content.split("Abstract:")[1].strip()
            else:
                abstract = content
            
            # Combine title and abstract for better embedding
            combined_text = f"{title}. {abstract}"
            texts.append(combined_text)
            file_names.append(file)
            titles.append(title)

print(f"Loaded {len(texts)} papers for embedding generation.\n")

if len(texts) > 0:
    print("Generating embeddings (this may take a few minutes)...")
    embeddings = model.encode(texts, show_progress_bar=True, convert_to_numpy=True)
    
    np.save(os.path.join(OUTPUT_DIR, "embeddings.npy"), embeddings)
    
    with open(os.path.join(OUTPUT_DIR, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(
            [{"filename": fn, "title": t, "text": txt} 
             for fn, t, txt in zip(file_names, titles, texts)],
            f, ensure_ascii=False, indent=2
        )
    
    print(f"✅ Saved embeddings and metadata inside {OUTPUT_DIR}\n")
    
    # -----------------------------
    # STEP 3: Clustering
    # -----------------------------
    # Adjust number of clusters based on papers count
    actual_clusters = min(N_CLUSTERS, max(2, len(texts) // 20))
    
    print(f"Running KMeans clustering with {actual_clusters} clusters...")
    kmeans = KMeans(n_clusters=actual_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(embeddings)
    
    with open(os.path.join(OUTPUT_DIR, "cluster_assignments.csv"), "w", encoding="utf-8") as f:
        f.write("filename,cluster,title\n")
        for fn, lbl, t in zip(file_names, labels, titles):
            # Escape title for CSV
            t_escaped = t.replace('"', '""')
            f.write(f'{fn},{lbl},"{t_escaped}"\n')
    
    print(f"✅ Clustering completed and saved to cluster_assignments.csv\n")
    
    # -----------------------------
    # STEP 4: Find representative paper per cluster
    # -----------------------------
    print("Finding representative papers and analyzing clusters...")
    representatives = []
    cluster_stats = []
    
    for i in range(actual_clusters):
        cluster_indices = np.where(labels == i)[0]
        cluster_vectors = embeddings[cluster_indices]
        centroid = kmeans.cluster_centers_[i]
        sims = cosine_similarity([centroid], cluster_vectors)[0]
        best_idx = cluster_indices[np.argmax(sims)]
        
        representatives.append((i, file_names[best_idx], titles[best_idx], texts[best_idx]))
        cluster_stats.append({
            "cluster_id": i,
            "size": len(cluster_indices),
            "representative": file_names[best_idx],
            "representative_title": titles[best_idx][:100]
        })
    
    with open(os.path.join(OUTPUT_DIR, "representatives.txt"), "w", encoding="utf-8") as f:
        for cid, fname, title, text in representatives:
            f.write("=" * 80 + "\n")
            f.write(f"Cluster {cid} (Size: {cluster_stats[cid]['size']} papers)\n")
            f.write(f"Representative File: {fname}\n")
            f.write(f"Title: {title}\n")
            f.write(f"Abstract Preview:\n{text[:500]}...\n\n")
    
    # Save cluster statistics
    with open(os.path.join(OUTPUT_DIR, "cluster_stats.json"), "w", encoding="utf-8") as f:
        json.dump(cluster_stats, f, indent=2)
    
    print("✅ Representative papers saved to representatives.txt")
    print("✅ Cluster statistics saved to cluster_stats.json")
    
    # Final summary
    print("\n" + "=" * 60)
    print("FINAL SUMMARY")
    print("=" * 60)
    print(f"Total papers downloaded: {downloaded_count}")
    print(f"Date range: {START_DATE.date()} to {END_DATE.date()}")
    print(f"Number of clusters: {actual_clusters}")
    print(f"Output directory: {OUTPUT_DIR}/")
    print("\nGenerated files:")
    print("  - PDFs and text files for each paper")
    print("  - papers_metadata.json (detailed paper info)")
    print("  - embeddings.npy (embedding vectors)")
    print("  - meta.json (embedding metadata)")
    print("  - cluster_assignments.csv (cluster assignments)")
    print("  - representatives.txt (cluster representatives)")
    print("  - cluster_stats.json (cluster statistics)")
    
    # Show cluster distribution
    print(f"\nCluster size distribution:")
    for stat in sorted(cluster_stats, key=lambda x: x['size'], reverse=True):
        print(f"  Cluster {stat['cluster_id']}: {stat['size']} papers - {stat['representative_title'][:60]}...")
    
else:
    print("No papers found matching the criteria.")

print("\n🎯 Script completed successfully!")