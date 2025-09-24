import os
import requests
import feedparser
from datetime import datetime

# Fixed search query
topic_query = "uncertainty prediction ML"

# Date filter for September 2025
START_DATE = datetime(2025, 9, 1)
END_DATE = datetime(2025, 9, 30, 23, 59, 59)

# Output folder
OUTPUT_DIR = "arxiv_papers_sept2025"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Base URL for arXiv API
base_url = "http://export.arxiv.org/api/query"

# Function to fetch papers from arXiv
def fetch_papers(query, max_results=100):
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

# Function to check if paper is from September 2025
def is_september_2025(date_str):
    date = datetime.strptime(date_str, "%Y-%m-%dT%H:%M:%SZ")
    return START_DATE <= date <= END_DATE

# Function to download a file
def download_file(url, filepath):
    r = requests.get(url, stream=True)
    r.raise_for_status()
    with open(filepath, "wb") as f:
        for chunk in r.iter_content(chunk_size=8192):
            f.write(chunk)

# Main execution
if __name__ == "__main__":
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

        # Find PDF link
        for link in entry.links:
            if link.rel == "related" and "doi.org" in link.href:
                doi = link.href
            if link.get("title") == "pdf":
                pdf_url = link.href

        if not pdf_url:
            continue

        # Create safe filename
        arxiv_id = entry.id.split("/")[-1]
        pdf_filename = os.path.join(OUTPUT_DIR, f"{arxiv_id}.pdf")
        txt_filename = os.path.join(OUTPUT_DIR, f"{arxiv_id}.txt")

        # Download PDF
        if not os.path.exists(pdf_filename):
            print(f"Downloading PDF: {title}")
            download_file(pdf_url, pdf_filename)

        # Save metadata in txt file
        with open(txt_filename, "w", encoding="utf-8") as f:
            f.write(f"Title: {title}\n")
            f.write(f"Published: {published}\n")
            f.write(f"DOI/ID: {doi}\n")
            f.write("Abstract:\n")
            f.write(abstract + "\n")

    print("Download completed. Check the folder:", OUTPUT_DIR)
