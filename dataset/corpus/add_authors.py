"""Add authors and publication histories to target papers."""
import os
import argparse
from tqdm import tqdm

import utils


def download_and_attach_authors_to_papers(all_papers_dict, role, args):
    """Fetch author rosters from S2 API and attach to papers with the given role."""
    corpus_ids = [cid for cid, p in all_papers_dict.items() if role in p["roles"]]
    utils.log(f"Fetching author rosters for {len(corpus_ids)} {role} papers")

    s2_ids = [f"CorpusId:{cid}" for cid in corpus_ids]
    records = utils.s2_batch_lookup(
        s2_ids,
        url=f"{utils.S2_API_BASE}/paper/batch",
        fields=["corpusId", "authors"],
        batch_size=args.batch_size,
    )

    corpus_id_to_authors = {}
    for record in records:
        if record is None or record.get("corpusId") is None or not record.get("authors"):
            continue
        corpus_id = str(record["corpusId"])
        authors = []
        for author_data in record["authors"]:
            if author_data.get("authorId") is None:
                continue
            authors.append({
                "author_id": str(author_data["authorId"]),
                "name": author_data.get("name", ""),
            })
        if authors:
            corpus_id_to_authors[corpus_id] = authors

    distinct_author_ids = {a["author_id"] for authors in corpus_id_to_authors.values() for a in authors}
    utils.log(f"Found {len(distinct_author_ids)} distinct authors across {len(corpus_id_to_authors)} {role} papers ({len(corpus_ids) - len(corpus_id_to_authors)} papers have no authors)")

    for corpus_id, authors in corpus_id_to_authors.items():
        if corpus_id in all_papers_dict and "authors" not in all_papers_dict[corpus_id]:
            all_papers_dict[corpus_id]["authors"] = authors

    return all_papers_dict


def download_publication_histories_for_target_authors(all_papers_dict, args):
    """Fetch publication histories for target authors from S2 batch API."""
    all_author_ids = set()
    for paper in all_papers_dict.values():
        if "target" in paper["roles"] and "authors" in paper:
            for author in paper["authors"]:
                all_author_ids.add(author["author_id"])

    utils.log(f"Fetching publication histories for {len(all_author_ids)} distinct target paper authors")

    records = utils.s2_batch_lookup(
        list(all_author_ids),
        url=f"{utils.S2_API_BASE}/author/batch",
        fields=["authorId", "papers", "papers.corpusId"],
        batch_size=100,
    )

    author_id_to_corpus_ids = {}
    for record in records:
        if record is None or record.get("authorId") is None:
            continue
        author_id = str(record["authorId"])
        papers = record.get("papers") or []
        corpus_ids = [str(p["corpusId"]) for p in papers if p and p.get("corpusId")]
        if corpus_ids:
            author_id_to_corpus_ids[author_id] = corpus_ids

    total_pubs = sum(len(pubs) for pubs in author_id_to_corpus_ids.values())
    utils.log(f"Found {total_pubs} publication corpus IDs for {len(author_id_to_corpus_ids)} authors ({len(all_author_ids) - len(author_id_to_corpus_ids)} authors have no publications)")
    return author_id_to_corpus_ids


def download_publication_history_papers(author_id_to_corpus_ids, args):
    """Download publication history paper metadata from S2 API."""
    all_pub_corpus_ids = {cid for cids in author_id_to_corpus_ids.values() for cid in cids}
    utils.log(f"Downloading metadata for {len(all_pub_corpus_ids)} unique publication history papers")

    s2_ids = [f"CorpusId:{cid}" for cid in all_pub_corpus_ids]
    records = utils.s2_batch_lookup(
        s2_ids,
        url=f"{utils.S2_API_BASE}/paper/batch",
        fields=["corpusId", "externalIds", "title", "abstract", "publicationDate"],
        batch_size=args.batch_size,
    )

    pub_history_dict = {}
    for record in records:
        if record is None or record.get("corpusId") is None:
            continue
        arxiv_id = (record.get("externalIds") or {}).get("ArXiv")
        if arxiv_id is None:
            continue
        date = record.get("publicationDate")
        if date is None:
            continue
        corpus_id = str(record["corpusId"])
        pub_history_dict[corpus_id] = {
            "corpus_id": corpus_id,
            "arxiv_id": arxiv_id,
            "date": date,
            "title": (record.get("title") or "").strip(),
            "abstract": (record.get("abstract") or "").strip(),
            "categories": [],  # populated in Stage 6 from arXiv snapshot
            "roles": ["target.author.publication_history"],
        }

    utils.log(f"Found {len(pub_history_dict)} publication history papers with arXiv IDs (out of {len(all_pub_corpus_ids)} total)")

    # Refine author_id_to_corpus_ids to only include papers we found
    total_before = sum(len(pubs) for pubs in author_id_to_corpus_ids.values())
    for author_id in author_id_to_corpus_ids:
        author_id_to_corpus_ids[author_id] = [
            cid for cid in author_id_to_corpus_ids[author_id]
            if cid in pub_history_dict
        ]
    total_after = sum(len(pubs) for pubs in author_id_to_corpus_ids.values())
    utils.log(f"Refined author publication lists from {total_before} to {total_after} entries with available abstracts")

    return author_id_to_corpus_ids, pub_history_dict


def attach_and_insert_publication_histories(all_papers_dict, author_id_to_corpus_ids, pub_history_dict):
    """Attach publication histories to target authors and insert papers into corpus."""
    utils.log("Attaching publication histories to target paper authors")

    total_attached = 0
    authors_with_empty_history = 0
    for paper in tqdm(all_papers_dict.values(), desc="Attaching publication histories"):
        if "target" in paper["roles"]:
            for author in paper["authors"]:
                author_id = author["author_id"]
                corpus_ids = author_id_to_corpus_ids.get(author_id, [])
                valid_corpus_ids = [
                    corpus_id for corpus_id in corpus_ids
                    if pub_history_dict[corpus_id]["date"] < paper["date"]
                ]
                valid_corpus_ids.sort(key=lambda corpus_id: pub_history_dict[corpus_id]["date"])
                author["publication_history"] = valid_corpus_ids
                total_attached += len(valid_corpus_ids)
                if len(valid_corpus_ids) == 0:
                    authors_with_empty_history += 1

    utils.log(f"Attached {total_attached} publication history entries ({authors_with_empty_history} authors have empty histories due to date truncation or no prior publications)")

    # Insert publication history papers into corpus
    corpus_size_before = len(all_papers_dict)
    for corpus_id, paper in pub_history_dict.items():
        if corpus_id in all_papers_dict:
            if "target.author.publication_history" not in all_papers_dict[corpus_id]["roles"]:
                all_papers_dict[corpus_id]["roles"].append("target.author.publication_history")
        else:
            all_papers_dict[corpus_id] = paper

    utils.log(f"Inserted {len(all_papers_dict) - corpus_size_before} new publication history papers into corpus (corpus now contains {len(all_papers_dict)} papers)")
    return all_papers_dict


def download_and_attach_key_references_and_prune(all_papers_dict, args):
    """Download key references for publication history papers, attach them, and prune."""
    pub_history_corpus_ids = [
        cid for cid, p in all_papers_dict.items()
        if "target.author.publication_history" in p["roles"] and "key_references" not in p
    ]
    utils.log(f"Fetching key references for {len(pub_history_corpus_ids)} publication history papers")

    records = utils.s2_fetch_references(
        pub_history_corpus_ids,
        fields=["corpusId", "externalIds", "title", "abstract", "publicationDate"],
    )

    # Build mapping from corpus_id to its batch record for easy lookup
    batch_results = {}
    for record in records:
        if record is not None and record.get("corpusId") is not None:
            batch_results[str(record["corpusId"])] = record

    key_ref_papers_dict = {}
    for corpus_id in pub_history_corpus_ids:
        record = batch_results.get(corpus_id)
        if record is None:
            all_papers_dict[corpus_id]["key_references"] = []
            continue

        paper_date = all_papers_dict[corpus_id]["date"]
        refs = record.get("references") or []
        influential_refs = []
        for ref in refs:
            if not ref.get("isInfluential"):
                continue
            if ref.get("corpusId") is None:
                continue
            cited_id = str(ref["corpusId"])
            arxiv_id = (ref.get("externalIds") or {}).get("ArXiv")
            if arxiv_id is None:
                continue
            cited_date = ref.get("publicationDate")
            if cited_date is None or cited_date >= paper_date:
                continue
            influential_refs.append(cited_id)
            if cited_id not in key_ref_papers_dict:
                key_ref_papers_dict[cited_id] = {
                    "corpus_id": cited_id,
                    "arxiv_id": arxiv_id,
                    "date": cited_date,
                    "title": (ref.get("title") or "").strip(),
                    "abstract": (ref.get("abstract") or "").strip(),
                    "categories": [],
                    "roles": [],
                }

        influential_refs = list(set(influential_refs))
        all_papers_dict[corpus_id]["key_references"] = [{"corpus_id": ref_id} for ref_id in influential_refs]

    total_refs = sum(len(all_papers_dict[cid]["key_references"]) for cid in pub_history_corpus_ids)
    num_with_refs = sum(1 for cid in pub_history_corpus_ids if all_papers_dict[cid]["key_references"])
    utils.log(f"Found {total_refs} key references across {num_with_refs} / {len(pub_history_corpus_ids)} publication history papers")

    # Insert key reference papers into corpus
    corpus_size_before = len(all_papers_dict)
    for corpus_id, paper in key_ref_papers_dict.items():
        if corpus_id in all_papers_dict:
            if "target.author.publication_history.key_reference" not in all_papers_dict[corpus_id]["roles"]:
                all_papers_dict[corpus_id]["roles"].append("target.author.publication_history.key_reference")
        else:
            paper["roles"].append("target.author.publication_history.key_reference")
            all_papers_dict[corpus_id] = paper
    utils.log(f"Inserted {len(all_papers_dict) - corpus_size_before} new key reference papers (corpus now contains {len(all_papers_dict)} papers)")

    # Prune publication history papers exceeding max_key_references
    exceeding = {
        cid for cid in pub_history_corpus_ids
        if len(all_papers_dict[cid]["key_references"]) > args.max_key_references
    }
    if exceeding:
        all_papers_dict = {cid: p for cid, p in all_papers_dict.items() if cid not in exceeding}
        utils.log(f"Pruned {len(exceeding)} publication history papers exceeding {args.max_key_references} key references")
    all_papers_dict = utils.cleanup_after_pruning(all_papers_dict)

    return all_papers_dict


def prune_papers_without_authors_and_clean_references(all_papers_dict, role):
    """Remove papers of the given role that have no authors, then clean up references."""
    corpus_size_before = len(all_papers_dict)
    all_papers_dict = {
        corpus_id: paper for corpus_id, paper in all_papers_dict.items()
        if role not in paper["roles"] or ("authors" in paper and paper["authors"])
    }
    utils.log(f"Pruned {corpus_size_before - len(all_papers_dict)} {role} papers without authors (corpus now contains {len(all_papers_dict)} papers)")
    all_papers_dict = utils.cleanup_after_pruning(all_papers_dict)
    return all_papers_dict


def main():
    """Add authors and publication histories to target papers."""
    parser = argparse.ArgumentParser(description="Add authors and publication histories to corpus papers")
    parser.add_argument("--input_dir", type=str, default="data/corpus/test", help="Input directory containing stage02 papers")
    parser.add_argument("--output_dir", type=str, default="data/corpus/test", help="Output directory")
    parser.add_argument("--max_key_references", type=int, default=10, help="Max key references per pub history paper")
    parser.add_argument("--batch_size", type=int, default=500, help="Batch size for S2 API calls")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing output")
    args = parser.parse_args()

    input_path = os.path.join(args.input_dir, "all_papers.stage02.json")
    output_path = os.path.join(args.output_dir, "all_papers.stage03.json")

    utils.log(f"Loading {input_path}")
    all_papers, metadata = utils.load_json(input_path)
    all_papers_dict = {p["corpus_id"]: p for p in all_papers}

    # Add authors to target papers
    all_papers_dict = download_and_attach_authors_to_papers(all_papers_dict, "target", args)
    all_papers_dict = prune_papers_without_authors_and_clean_references(all_papers_dict, "target")

    # Download and attach publication histories
    author_id_to_corpus_ids = download_publication_histories_for_target_authors(all_papers_dict, args)
    author_id_to_corpus_ids, pub_history_dict = download_publication_history_papers(author_id_to_corpus_ids, args)
    all_papers_dict = attach_and_insert_publication_histories(all_papers_dict, author_id_to_corpus_ids, pub_history_dict)

    # Download and attach key references for publication history papers
    all_papers_dict = download_and_attach_key_references_and_prune(all_papers_dict, args)

    # Add authors to publication history papers
    all_papers_dict = download_and_attach_authors_to_papers(all_papers_dict, "target.author.publication_history", args)
    all_papers_dict = prune_papers_without_authors_and_clean_references(all_papers_dict, "target.author.publication_history")

    all_papers = list(all_papers_dict.values())
    utils.save_json(all_papers, output_path, utils.update_metadata(metadata, args), overwrite=args.overwrite)
    utils.log(f"Saved {len(all_papers)} papers to {output_path}")


if __name__ == "__main__":
    main()
