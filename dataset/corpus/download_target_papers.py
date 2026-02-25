"""Download target papers from local arXiv snapshot, S2 API keyword search, or a local JSONL file."""
import os
import argparse
import datetime

import utils

# Default arXiv categories covering forestry and related environmental/ecological fields.
# Most forestry research is not on arXiv; use --s2_keywords or --local_papers_path for
# direct Semantic Scholar keyword search or a pre-downloaded JSONL corpus instead.
FORESTRY_ARXIV_CATEGORIES = [
    "q-bio.PE",   # Quantitative Biology: Populations and Evolution (ecology, conservation)
    "eess.SP",    # Signal Processing (remote sensing of forests)
    "cs.CV",      # Computer Vision (forest/vegetation detection from imagery)
    "eess.IV",    # Image and Video Processing (satellite/aerial forest imagery)
]

# Semantic Scholar keyword query used when --s2_keywords flag is set.
FORESTRY_S2_KEYWORDS = (
    "forestry OR silviculture OR \"forest ecology\" OR \"forest management\" "
    "OR \"forest carbon\" OR \"forest biomass\" OR \"wildfire\" OR \"forest fire\" "
    "OR \"deforestation\" OR \"reforestation\" OR \"forest inventory\" "
    "OR \"tree species\" OR \"forest biodiversity\""
)


def parse_arxiv_created_date(date_str):
    """Parse arXiv version created date (RFC 2822 format) to YYYY-MM-DD string."""
    dt = datetime.datetime.strptime(date_str.strip(), "%a, %d %b %Y %H:%M:%S %Z")
    return dt.strftime("%Y-%m-%d")


def filter_arxiv_snapshot(arxiv_snapshot, start_date, end_date, categories):
    """Filter arXiv snapshot to papers in target categories and date range."""
    categories_lower = {c.lower() for c in categories}
    target_papers = []
    for arxiv_id, record in arxiv_snapshot.items():
        paper_categories = record["categories"].strip().split()
        if not any(cat.lower() in categories_lower for cat in paper_categories):
            continue
        date = parse_arxiv_created_date(record["versions"][0]["created"])
        if date < start_date or date >= end_date:
            continue
        target_papers.append({
            "arxiv_id": arxiv_id,
            "date": date,
            "categories": paper_categories,
            "title": record["title"].strip().replace("\n", " "),
            "abstract": record["abstract"].strip().replace("\n", " "),
        })
    utils.log(f"Filtered {len(target_papers)} target papers from arXiv snapshot")
    return target_papers


def search_s2_by_keywords(keywords, start_date, end_date, max_results=10000):
    """Search Semantic Scholar for forestry papers by keyword query and date range.

    Returns a list of paper dicts with corpus_id, title, abstract, and date already set.
    Papers returned here skip the separate lookup_corpus_ids step.
    """
    import time
    import requests

    s2_api_key = utils._get_s2_api_key()
    fields = "corpusId,title,abstract,publicationDate,externalIds"
    url = f"{utils.S2_API_BASE}/paper/search/bulk"
    params = {
        "query": keywords,
        "fields": fields,
        "publicationDateOrYear": f"{start_date}:{end_date}",
        "limit": 1000,
    }
    headers = {"x-api-key": s2_api_key}

    papers = []
    token = None
    while True:
        if token:
            params["token"] = token
        for attempt in range(5):
            try:
                resp = requests.get(url, params=params, headers=headers, timeout=30)
                resp.raise_for_status()
                data = resp.json()
                break
            except Exception as e:
                utils.log(f"S2 keyword search error (attempt {attempt + 1}/5): {e}")
                time.sleep(10)
        else:
            utils.log("S2 keyword search failed after 5 retries; stopping early")
            break

        for item in data.get("data", []):
            if item is None or not item.get("corpusId") or not item.get("title") or not item.get("abstract"):
                continue
            pub_date = (item.get("publicationDate") or "")[:10]
            if not pub_date or pub_date < start_date or pub_date >= end_date:
                continue
            arxiv_id = (item.get("externalIds") or {}).get("ArXiv")
            papers.append({
                "corpus_id": str(item["corpusId"]),
                "arxiv_id": arxiv_id,
                "date": pub_date,
                "categories": [],
                "title": item["title"].strip().replace("\n", " "),
                "abstract": item["abstract"].strip().replace("\n", " "),
                "roles": ["target"],
            })
            if len(papers) >= max_results:
                utils.log(f"Reached max_results={max_results}; stopping keyword search")
                return papers

        token = data.get("token")
        utils.log(f"Retrieved {len(papers)} papers so far (token={'present' if token else 'none'})")
        if not token:
            break
        time.sleep(1.5)

    utils.log(f"S2 keyword search returned {len(papers)} papers")
    return papers


def load_local_papers(local_papers_path, start_date, end_date):
    """Load pre-downloaded forestry papers from a JSONL file and filter by date range.

    Each line must be a JSON object with at minimum: corpus_id, title, abstract, date.
    """
    import json
    papers = []
    with open(local_papers_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            date = rec.get("date", "")[:10]
            if not date or date < start_date or date >= end_date:
                continue
            if not rec.get("corpus_id") or not rec.get("title") or not rec.get("abstract"):
                continue
            rec.setdefault("roles", ["target"])
            rec.setdefault("categories", [])
            rec.setdefault("arxiv_id", None)
            papers.append(rec)
    utils.log(f"Loaded {len(papers)} papers from local file {local_papers_path}")
    return papers


def lookup_corpus_ids(target_papers):
    """Look up S2 corpus IDs for arXiv papers via batch API."""
    s2_ids = [f"ARXIV:{p['arxiv_id']}" for p in target_papers]
    records = utils.s2_batch_lookup(
        s2_ids,
        url=f"{utils.S2_API_BASE}/paper/batch",
        fields=["corpusId", "externalIds"],
        batch_size=500,
    )
    arxiv_to_corpus = {}
    for record in records:
        if record is None or record.get("corpusId") is None:
            continue
        external_ids = record.get("externalIds") or {}
        if external_ids.get("ArXiv") is None:
            continue
        arxiv_to_corpus[external_ids["ArXiv"]] = str(record["corpusId"])
    return arxiv_to_corpus


def main():
    parser = argparse.ArgumentParser(
        "Download forestry target papers from arXiv snapshot, S2 keyword search, or a local JSONL file"
    )
    parser.add_argument("--arxiv_snapshot_path", type=str, default="data/arxiv_snapshot/arxiv-metadata-oai-snapshot.json", help="Path to arXiv metadata snapshot (used when --s2_keywords and --local_papers_path are not set)")
    parser.add_argument("--start_date", type=str, default="2020-01-01", help="Start date for target papers (YYYY-MM-DD)")
    parser.add_argument("--end_date", type=str, default="2025-01-01", help="End date for target papers (YYYY-MM-DD)")
    parser.add_argument("--categories", nargs="+", default=FORESTRY_ARXIV_CATEGORIES, help="arXiv categories to include (used with arXiv snapshot mode)")
    parser.add_argument("--s2_keywords", action="store_true", help="Search Semantic Scholar by forestry keywords instead of using an arXiv snapshot (requires S2_API_KEY)")
    parser.add_argument("--s2_query", type=str, default=FORESTRY_S2_KEYWORDS, help="Custom keyword query for S2 search (only used with --s2_keywords)")
    parser.add_argument("--s2_max_results", type=int, default=10000, help="Maximum number of papers to retrieve via S2 keyword search")
    parser.add_argument("--local_papers_path", type=str, default=None, help="Path to a JSONL file of pre-downloaded forestry papers (each line: JSON with corpus_id, title, abstract, date)")
    parser.add_argument("--output_dir", type=str, default="data/corpus/test", help="Output directory for target papers")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing output")
    args = parser.parse_args()

    if args.local_papers_path:
        utils.log(f"Loading papers from local file: {args.local_papers_path}")
        target_records = load_local_papers(args.local_papers_path, args.start_date, args.end_date)
    elif args.s2_keywords:
        utils.log(f"Searching Semantic Scholar with query: {args.s2_query}")
        target_records = search_s2_by_keywords(args.s2_query, args.start_date, args.end_date, args.s2_max_results)
    else:
        utils.log(f"Loading arXiv snapshot from: {args.arxiv_snapshot_path}")
        arxiv_snapshot = utils.load_arxiv_snapshot(args.arxiv_snapshot_path)
        target_papers = filter_arxiv_snapshot(arxiv_snapshot, args.start_date, args.end_date, args.categories)

        arxiv_to_corpus = lookup_corpus_ids(target_papers)
        utils.log(f"Found corpus IDs for {len(arxiv_to_corpus)} / {len(target_papers)} target papers")

        target_records = []
        for paper in target_papers:
            if paper["arxiv_id"] not in arxiv_to_corpus:
                continue
            paper["corpus_id"] = arxiv_to_corpus[paper["arxiv_id"]]
            paper["roles"] = ["target"]
            target_records.append(paper)

    target_records = sorted(target_records, key=lambda x: x["date"])
    all_papers_dict = {paper["corpus_id"]: paper for paper in target_records}
    all_papers_records = list(all_papers_dict.values())
    utils.log(f"Retained {len(all_papers_records)} unique target papers after deduplication")

    all_papers_path = os.path.join(args.output_dir, "all_papers.stage01.json")
    utils.save_json(all_papers_records, all_papers_path, utils.update_metadata([], args), overwrite=args.overwrite)


if __name__ == "__main__":
    main()
