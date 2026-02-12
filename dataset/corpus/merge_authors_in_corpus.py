import os
import argparse
from tqdm import tqdm

import utils
from dataset.s2and_prep.download_s2and_features import get_block

ILLEGAL_LASTNAMES = [
    "physics",
    "biology",
    "chemistry",
    "mathematics",
    "engineering",
    "computer",
    "science",
    "technology",
    "medicine",
    "health",
    "social",
    "humanities",
    "arts",
    "astronomy",
    "geology",
    "geography",
    "history",
    "psychology",
    "sociology",
    "political",
    "economics",
    "philosophy",
    "university",
    "institute",
    "research",
    "laboratory",
    "center",
    "college"
]


def render_new_sd_author_id(author_record, sd2og):
    """Create a new sd_author_id for a newly discovered author."""
    if "first_name" in author_record:
        block_id = get_block(author_record)
    elif "name" in author_record:
        parts = author_record["name"].split(" ")
        first_name = parts[0] if parts else ""
        last_name = parts[-1] if len(parts) > 1 else ""
        temp_author_record = {
            "first_name": first_name,
            "last_name": last_name
        }
        block_id = get_block(temp_author_record)
    else:
        raise ValueError("Author record lacks both 'first_name' and 'name' fields.")

    idx = 0
    while True:
        candidate_sd_author_id = f"{block_id}_{idx}"
        if candidate_sd_author_id not in sd2og:
            return candidate_sd_author_id
        idx += 1


def load_s2and_clusters(results_dir):
    """Load every S2AND cluster pickle from results_dir and return merged dict."""
    utils.log(f"Loading S2AND clusters from {results_dir}")
    cluster_files = []
    for filename in os.listdir(results_dir):
        if filename.endswith(".pickle") and "clusters" in filename:
            cluster_files.append(filename)
    utils.log(f"Found {len(cluster_files)} cluster pickle files")
    sd_authors = {}
    for filename in tqdm(cluster_files, desc="Loading S2AND clusters"):
        data, _ = utils.load_pkl(os.path.join(results_dir, filename))
        for cluster_key, signatures in data.items():
            if cluster_key in sd_authors:
                raise ValueError(f"Duplicate cluster key found: {cluster_key}")
            sd_authors[cluster_key] = signatures
    utils.log(f"Loaded {len(sd_authors)} clusters from S2AND results")
    return sd_authors


def is_stopword_cluster(sd):
    """Return True if the cluster label appears to be a stopword artifact."""
    parts = sd.split(" ")
    firstname = parts[0] if parts else ""
    lastname = parts[1].split("_")[0] if len(parts) > 1 else ""
    return (firstname == "_") or any(illegal in lastname for illegal in ILLEGAL_LASTNAMES)


def prune_stopword_clusters(sd2og, og2sd, sd2publications):
    """Drop clusters whose labels look like stopwords and return filtered mappings."""
    allowed_sds = set()
    for sd in sd2og:
        if not is_stopword_cluster(sd):
            allowed_sds.add(sd)
    filtered_sd2og = {sd: ogs for sd, ogs in sd2og.items() if sd in allowed_sds}
    filtered_og2sd = {og: sd for og, sd in og2sd.items() if sd in allowed_sds}
    filtered_sd2publications = {sd: pubs for sd, pubs in sd2publications.items() if sd in allowed_sds}
    pruned_authors_sd = set(sd2og.keys()) - set(filtered_sd2og.keys())
    pruned_authors_og = set(og2sd.keys()) - set(filtered_og2sd.keys())
    utils.log(f"Pruned {len(pruned_authors_sd)} stopword clusters out of {len(sd2og)}")
    return filtered_sd2og, filtered_og2sd, filtered_sd2publications, pruned_authors_sd, pruned_authors_og


def build_author_mappings(sd_authors, all_papers_dict):
    """Create og2sd/sd2og mappings plus raw publication lists from S2AND clusters."""
    sd2og = {}
    og2sd = {}
    sd2publications = {}
    duplicate_og_ids = 0
    for sd, cluster in tqdm(sd_authors.items(), desc="Building author mappings"):
        for signature in cluster:
            paper_id, og_author_id = signature.split("_")

            # populate og2sd, sd2og, and sd2publications consistently
            # about 2.5% of OG IDs appear in multiple S2AND clusters - we keep first assignment
            if og_author_id not in og2sd:
                # new OG ID - add to all three mappings
                og2sd[og_author_id] = sd
                if sd not in sd2og:
                    sd2og[sd] = set()
                sd2og[sd].add(og_author_id)
                if sd not in sd2publications:
                    sd2publications[sd] = set()
                sd2publications[sd].add(paper_id)
            elif og2sd[og_author_id] == sd:
                # same cluster - add paper to publications
                if sd not in sd2publications:
                    sd2publications[sd] = set()
                sd2publications[sd].add(paper_id)
            else:
                # different cluster - duplicate OG ID, skip
                duplicate_og_ids += 1

    if duplicate_og_ids > 0:
        utils.log(f"Warning: {duplicate_og_ids} OG author IDs appeared in multiple S2AND clusters (kept first assignment)")

    # S2AND fails to return clusters for a few (O(10)) authors. Let's just assume they remain singletons and put them in manually.
    utils.log("Adding singleton authors missing from S2AND clusters")
    hotfixed_singleton_authors = set()
    for paper in tqdm(all_papers_dict.values(), desc="Scanning for authors missing from S2AND clusters and hotfixing as singletons"):
        if "authors" in paper: # this will be true for "target" ∪ "target.author.publication_history" papers
            for author in paper["authors"]:
                if author["author_id"] not in og2sd:
                    hotfixed_singleton_authors.add(author["author_id"])
                    og_author_id = author["author_id"]
                    sd_author_id = render_new_sd_author_id(author, sd2og)
                    og2sd[og_author_id] = sd_author_id
                    sd2og[sd_author_id] = {og_author_id}

    for paper in tqdm(all_papers_dict.values(), desc="Adding hotfixed singleton authors to sd2publications"):
        if "authors" in paper: # this will be true for "target" ∪ "target.author.publication_history" papers
            for author in paper["authors"]:
                og_author_id = author["author_id"]
                if og_author_id in hotfixed_singleton_authors:
                    sd_author_id = og2sd[og_author_id]
                    if sd_author_id not in sd2publications:
                        sd2publications[sd_author_id] = set()
                    sd2publications[sd_author_id].add(paper["corpus_id"])

    utils.log(f"Hotfixed {len(hotfixed_singleton_authors)} singleton authors missing from S2AND clusters")

    sd2og, og2sd, sd2publications, pruned_authors_sd, pruned_authors_og = prune_stopword_clusters(sd2og, og2sd, sd2publications)
    utils.log(f"Constructed mappings for {len(og2sd)} original authors across {len(sd2og)} merged clusters")
    return sd2og, og2sd, sd2publications, pruned_authors_sd, pruned_authors_og

def prune_papers_with_illegal_authors_from_dataset(all_papers_dict, pruned_authors_og):
    """Remove papers by any illegal authors (and all their descendants) from all_papers_dict."""
    pruned_corpus_ids = set()
    for corpus_id, paper in tqdm(all_papers_dict.items(), desc="Scanning for illegal papers"):
        if "authors" in paper: # this will be true for "target" ∪ "target.author.publication_history" papers
            for author in paper["authors"]:
                og_author_id = author["author_id"]
                if og_author_id in pruned_authors_og:
                    pruned_corpus_ids.add(corpus_id)
                    break

    # Remove obviously illegal papers from all_papers_dict
    all_papers_dict = {corpus_id: paper for corpus_id, paper in all_papers_dict.items() if corpus_id not in pruned_corpus_ids}
    utils.log(f"Pruned {len(pruned_corpus_ids)} papers with illegal authors from dataset")

    all_papers_dict = utils.cleanup_after_pruning(all_papers_dict)
    utils.log(f"Retained {len(all_papers_dict)} papers after pruning illegal authors and cleaning up")
    return all_papers_dict


def download_missing_target_author_pub_history_papers_and_prune(og2sd, sd2publications, all_papers_dict, args):
    """Download missing target authors' publication history papers from S2 API, tag and insert them, and update sd2publications. Also prunes unreachable papers after date truncation."""

    # Identify missing publications
    missing_corpus_ids = set()
    for paper in tqdm(all_papers_dict.values(), desc="Identifying missing target author publication history papers"):
        if "target" in paper["roles"]:
            for author in paper["authors"]:
                og_author_id = author["author_id"]
                sd_author_id = og2sd[og_author_id]
                author_publications = sd2publications[sd_author_id]
                for pub_corpus_id in author_publications:
                    if pub_corpus_id not in all_papers_dict:
                        missing_corpus_ids.add(pub_corpus_id)
                    elif "target.author.publication_history" not in all_papers_dict[pub_corpus_id]["roles"]:
                        all_papers_dict[pub_corpus_id]["roles"].append("target.author.publication_history")
    utils.log(f"Identified {len(missing_corpus_ids)} missing publications")

    # Download missing papers from S2 API
    s2_ids = [f"CorpusId:{cid}" for cid in missing_corpus_ids]
    records = utils.s2_batch_lookup(
        s2_ids,
        url=f"{utils.S2_API_BASE}/paper/batch",
        fields=["corpusId", "externalIds", "title", "abstract", "publicationDate"],
        batch_size=args.batch_size,
    )
    downloaded_papers_records = utils.parse_s2_paper_records(records)
    downloaded_papers_records = utils.add_role(downloaded_papers_records, "target.author.publication_history")
    utils.log(f"Downloaded {len(downloaded_papers_records)} missing publications with arXiv versions from S2 API")

    # merge downloaded papers into all_papers_dict
    downloaded_papers_dict = {paper["corpus_id"]: paper for paper in downloaded_papers_records}
    all_papers_dict = utils.union_records(all_papers_dict, downloaded_papers_dict)
    utils.log(f"Total number of papers after merging: {len(all_papers_dict)}")

    # Replace old publication histories with merged sd2publications, filtered by corpus membership and date
    utils.log("Attaching merged publication histories from sd2publications to target authors")
    for paper in tqdm(all_papers_dict.values(), desc="Attaching merged publication histories"):
        if "target" in paper["roles"]:
            for author in paper["authors"]:
                og_author_id = author["author_id"]
                sd_author_id = og2sd[og_author_id]
                merged_publications = sd2publications[sd_author_id]
                # Filter to only include papers that exist in corpus and predate this target paper
                revised_publication_history = [c for c in merged_publications if (c in all_papers_dict) and (all_papers_dict[c]["date"] < paper["date"])]
                author["publication_history"] = revised_publication_history

    # prune now unreachable papers from all_papers_dict
    all_papers_dict = utils.remove_unreachable_papers(all_papers_dict)
    utils.log(f"Retained {len(all_papers_dict)} papers after removing unreachable papers")

    utils.log("Revising sd2publications to exclude non-arxiv papers and future papers")
    for sd, publications in sd2publications.items():
        sd2publications[sd] = {pub for pub in publications if pub in all_papers_dict}

    return all_papers_dict, sd2publications

def download_missing_target_author_pub_history_key_references_papers_and_prune(all_papers_dict, args):
    """Download key references for publication history papers missing them from S2 API, insert referenced papers, prune papers exceeding max_key_references, and clean up the corpus."""

    # Identify corpus_ids of target.author.publication_history papers missing key_references
    incomplete_key_reference_corpus_ids = [corpus_id for corpus_id, paper in all_papers_dict.items() if \
        ("target.author.publication_history" in paper["roles"]) and ("key_references" not in paper)
    ]
    utils.log(f"Identified {len(incomplete_key_reference_corpus_ids)} target.author.publication_history papers missing key_references")

    # Fetch key references from S2 batch API (with isInfluential filter)
    s2_ids = [f"CorpusId:{cid}" for cid in incomplete_key_reference_corpus_ids]
    records = utils.s2_batch_lookup(
        s2_ids,
        url=f"{utils.S2_API_BASE}/paper/batch",
        fields=["corpusId", "references", "references.isInfluential", "references.corpusId",
                "references.externalIds", "references.title", "references.abstract", "references.publicationDate"],
        batch_size=args.batch_size,
    )

    # Build mapping from corpus_id to its batch record for easy lookup
    batch_results = {}
    for record in records:
        if record is not None and record.get("corpusId") is not None:
            batch_results[str(record["corpusId"])] = record

    key_ref_papers_dict = {}
    for corpus_id in incomplete_key_reference_corpus_ids:
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

    total_refs = sum(len(all_papers_dict[cid]["key_references"]) for cid in incomplete_key_reference_corpus_ids)
    num_with_refs = sum(1 for cid in incomplete_key_reference_corpus_ids if all_papers_dict[cid]["key_references"])
    utils.log(f"Found {total_refs} key references across {num_with_refs} / {len(incomplete_key_reference_corpus_ids)} publication history papers")

    # Insert key reference papers into corpus (merge to preserve existing fields)
    key_ref_papers_records = list(key_ref_papers_dict.values())
    key_ref_papers_records = utils.add_role(key_ref_papers_records, "target.author.publication_history.key_reference")
    key_ref_papers_dict = {paper["corpus_id"]: paper for paper in key_ref_papers_records}
    corpus_size_before = len(all_papers_dict)
    all_papers_dict = utils.union_records(all_papers_dict, key_ref_papers_dict)
    utils.log(f"Inserted {len(all_papers_dict) - corpus_size_before} new key reference papers (corpus now contains {len(all_papers_dict)} papers)")

    # Prune publication history papers exceeding max_key_references
    pub_history_corpus_ids = {
        corpus_id for corpus_id, paper in all_papers_dict.items()
        if "target.author.publication_history" in paper["roles"]
    }
    pub_history_before = len(pub_history_corpus_ids)
    exceeding_corpus_ids = {
        corpus_id for corpus_id in pub_history_corpus_ids
        if len(all_papers_dict[corpus_id]["key_references"]) > args.max_key_references
    }
    all_papers_dict = {
        corpus_id: paper for corpus_id, paper in all_papers_dict.items()
        if corpus_id not in exceeding_corpus_ids
    }
    utils.log(f"Pruned {len(exceeding_corpus_ids)} publication history papers exceeding {args.max_key_references} key references ({pub_history_before - len(exceeding_corpus_ids)} remain)")
    all_papers_dict = utils.cleanup_after_pruning(all_papers_dict)

    return all_papers_dict


def download_missing_target_author_pub_history_authors_info_and_prune(all_papers_dict, sd2og, og2sd, sd2publications, args):
    """Download missing author info for target.author.publication_history papers from S2 API, update the corresponding all_papers_dict entries, and update sd2og, og2sd, sd2publications accordingly. Also prunes publication history papers without authors and cleans up the corpus."""

    # Identify corpus_ids of target.author.publication_history papers missing author info
    incomplete_author_info_corpus_ids = [corpus_id for corpus_id, paper in all_papers_dict.items() if \
        ("target.author.publication_history" in paper["roles"]) and "authors" not in paper
    ]
    utils.log(f"Identified {len(incomplete_author_info_corpus_ids)} target.author.publication_history papers missing author info")

    # Download author info from S2 API
    s2_ids = [f"CorpusId:{cid}" for cid in incomplete_author_info_corpus_ids]
    records = utils.s2_batch_lookup(
        s2_ids,
        url=f"{utils.S2_API_BASE}/paper/batch",
        fields=["corpusId", "authors"],
        batch_size=args.batch_size,
    )

    # Build corpus_id -> authors mapping from batch response
    corpus_id_to_authors = {}
    all_downloaded_author_ids = set()
    for record in records:
        if record is None or record.get("corpusId") is None or not record.get("authors"):
            continue
        corpus_id = str(record["corpusId"])
        authors = []
        for author_data in record["authors"]:
            if author_data.get("authorId") is None:
                continue
            og_author_id = str(author_data["authorId"])
            authors.append({
                "author_id": og_author_id,
                "name": author_data.get("name", ""),
            })
            all_downloaded_author_ids.add(og_author_id)
        if authors:
            corpus_id_to_authors[corpus_id] = authors

    utils.log(f"Downloaded author info for {len(corpus_id_to_authors)} papers with {len(all_downloaded_author_ids)} distinct authors")
    new_authors_set = all_downloaded_author_ids - set(og2sd.keys())
    utils.log(f"Found {len(new_authors_set)} newly discovered authors not seen before")

    # Update sd2og, og2sd, sd2publications with any newly discovered authors
    utils.log("Updating sd2og, og2sd, sd2publications with newly downloaded author info")
    for corpus_id, authors in corpus_id_to_authors.items():
        for author in authors:
            og_author_id = author["author_id"]
            if og_author_id not in og2sd:
                sd_author_id = render_new_sd_author_id(author, sd2og)
                og2sd[og_author_id] = sd_author_id
                sd2og[sd_author_id] = {og_author_id}
                sd2publications[sd_author_id] = None

    # Update all_papers_dict with downloaded author info
    utils.log("Updating all_papers_dict with downloaded author info")
    for corpus_id, authors in corpus_id_to_authors.items():
        if corpus_id in all_papers_dict:
            all_papers_dict[corpus_id]["authors"] = authors

    # Prune publication history papers without authors
    pub_history_corpus_ids = {corpus_id for corpus_id, p in all_papers_dict.items() if "target.author.publication_history" in p["roles"]}
    no_author_pub_history = {corpus_id for corpus_id in pub_history_corpus_ids if "authors" not in all_papers_dict[corpus_id] or not all_papers_dict[corpus_id]["authors"]}
    if no_author_pub_history:
        all_papers_dict = {corpus_id: paper for corpus_id, paper in all_papers_dict.items() if corpus_id not in no_author_pub_history}
        utils.log(f"Pruned {len(no_author_pub_history)} publication history papers without authors ({len(pub_history_corpus_ids) - len(no_author_pub_history)} remain)")
    all_papers_dict = utils.cleanup_after_pruning(all_papers_dict)

    return all_papers_dict, sd2og, og2sd, sd2publications


def refine_sd2publications_after_pruning(sd2publications, all_papers_dict):
    """Iterate through sd2publications listings and remove entries missing from all_papers_dict or without authors."""

    utils.log("Refining sd2publications after pruning")
    total_sd2publications_removed = 0
    for sd, publications in sd2publications.items():
        if publications is not None:
            total_sd2publications_removed += len(sd2publications[sd])
            sd2publications[sd] = {
                pub for pub in publications
                if pub in all_papers_dict and "authors" in all_papers_dict[pub]
            }
            total_sd2publications_removed -= len(sd2publications[sd])
    utils.log(f"Removed {total_sd2publications_removed} sd2publications entries missing from all_papers_dict or without authors")

    return sd2publications


def replace_og_author_ids_with_sd_author_ids(all_papers_dict, og2sd):
    """Replace original author IDs with S2AND merged author IDs in all_papers_dict."""
    for paper in tqdm(all_papers_dict.values(), desc="Replacing original author IDs with S2AND merged author IDs"):
        if "authors" in paper: # this will be true for "target" ∪ "target.author.publication_history" papers
            for author in paper["authors"]:
                og_author_id = author["author_id"]
                sd_author_id = og2sd[og_author_id]
                author["author_id"] = sd_author_id
    return all_papers_dict



def sort_all_papers_and_publication_histories_by_date(all_papers_dict, sd2publications):
    """Sort all_papers_dict by date, and sort each author's publication history by date (in all_papers_dict and sd2publications)."""

    # Sort all_papers_dict by date
    utils.log("Sorting all_papers_dict by date")
    all_papers_dict = dict(sorted(all_papers_dict.items(), key=lambda item: item[1]["date"]))

    # Sort each author's publication history by date in all_papers_dict
    utils.log("Sorting each author's publication history by date in all_papers_dict")
    for paper in tqdm(all_papers_dict.values(), desc="Sorting publication histories in all_papers_dict"):
        if "target" in paper["roles"]:
            for author in paper["authors"]:
                author["publication_history"] = sorted(author["publication_history"], key=lambda x: all_papers_dict[x]["date"])

    # Sort each author's publication history by date in sd2publications
    utils.log("Sorting each author's publication history by date in sd2publications")
    for sd, publications in tqdm(sd2publications.items(), desc="Sorting publication histories in sd2publications"):
        if publications is not None:
            sorted_publications = sorted(publications, key=lambda x: all_papers_dict[x]["date"])
            sd2publications[sd] = sorted_publications

    return all_papers_dict, sd2publications



def main():
    """Entry point for the clean merge workflow."""
    parser = argparse.ArgumentParser("Clean author merge pipeline using S2AND clusters.")
    parser.add_argument("--input_dir", type=str, default="data/corpus/test", help="Directory containing all_papers.stage03.json")
    parser.add_argument("--s2and_results_dir", type=str, default="data/corpus/s2and_prescience", help="Directory containing S2AND clusters")
    parser.add_argument("--max_key_references", type=int, default=10, help="Max key references per publication history paper")
    parser.add_argument("--batch_size", type=int, default=500, help="Batch size for S2 API calls")
    parser.add_argument("--output_dir", type=str, default="data/corpus/test", help="Output directory for merged all_papers")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing output")
    args = parser.parse_args()

    all_papers, metadata = utils.load_json(os.path.join(args.input_dir, "all_papers.stage03.json"))
    metadata = metadata if metadata is not None else []
    all_papers_dict = {paper["corpus_id"]: paper for paper in all_papers}

    sd_authors = load_s2and_clusters(args.s2and_results_dir)
    sd2og, og2sd, sd2publications, pruned_authors_sd, pruned_authors_og = build_author_mappings(sd_authors, all_papers_dict)
    all_papers_dict = prune_papers_with_illegal_authors_from_dataset(all_papers_dict, pruned_authors_og)

    all_papers_dict, sd2publications = download_missing_target_author_pub_history_papers_and_prune(og2sd, sd2publications, all_papers_dict, args)
    all_papers_dict = download_missing_target_author_pub_history_key_references_papers_and_prune(all_papers_dict, args)
    all_papers_dict, sd2og, og2sd, sd2publications = download_missing_target_author_pub_history_authors_info_and_prune(all_papers_dict, sd2og, og2sd, sd2publications, args)

    sd2og, og2sd, sd2publications, pruned_authors_sd, pruned_authors_og = prune_stopword_clusters(sd2og, og2sd, sd2publications)
    all_papers_dict = prune_papers_with_illegal_authors_from_dataset(all_papers_dict, pruned_authors_og)
    sd2publications = refine_sd2publications_after_pruning(sd2publications, all_papers_dict)

    all_papers_dict = replace_og_author_ids_with_sd_author_ids(all_papers_dict, og2sd)

    all_papers_dict, sd2publications = sort_all_papers_and_publication_histories_by_date(all_papers_dict, sd2publications)
    sd2og = {sd: list(ogs) for sd, ogs in sd2og.items()}  # convert sets to lists for JSON serialization
    sd2publications = {sd: (None if pubs is None else list(pubs)) for sd, pubs in sd2publications.items()}  # convert sets to lists for JSON serialization

    all_papers = list(all_papers_dict.values())
    updated_metadata = utils.update_metadata(metadata, args)
    utils.log(f"Saving {len(all_papers)} papers with updated author merges to stage04 output")
    utils.save_json(all_papers, os.path.join(args.output_dir, "all_papers.stage04.json"), metadata=updated_metadata, overwrite=args.overwrite)
    utils.save_json(sd2og, os.path.join(args.output_dir, "sd2og.json"), metadata=updated_metadata, overwrite=args.overwrite)
    utils.save_json(og2sd, os.path.join(args.output_dir, "og2sd.json"), metadata=updated_metadata, overwrite=args.overwrite)
    utils.save_json(sd2publications, os.path.join(args.output_dir, "sd2publications.json"), metadata=updated_metadata, overwrite=args.overwrite)


if __name__ == "__main__":
    main()
