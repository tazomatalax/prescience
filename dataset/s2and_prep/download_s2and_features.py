"""Download S2AND features for corpus papers from S2 API."""
import os
import argparse
import numpy as np
from tqdm import tqdm

import utils


def get_block(author_record: dict):
    first_name_first_letter = "_" if ("first_name" not in author_record) or (author_record["first_name"] is None) or (len(author_record["first_name"]) == 0) else author_record["first_name"].lower()[0]
    last_name = "_" if ("last_name" not in author_record) or (author_record["last_name"] is None) or (len(author_record["last_name"]) == 0) else author_record["last_name"].lower()
    return first_name_first_letter + " " + last_name


def split_author_name(full_name):
    """Heuristically split a full author name into first, middle, last parts."""
    parts = full_name.strip().split()
    if len(parts) == 0:
        return "", None, ""
    if len(parts) == 1:
        return parts[0], None, ""
    first = parts[0]
    last = parts[-1]
    middle = parts[1] if len(parts) > 2 else None
    return first, middle, last


def main():
    parser = argparse.ArgumentParser(description="Download S2AND features for corpus papers from S2 API.")
    parser.add_argument("--input_dirs", type=str, nargs="+", default=["data/corpus/train", "data/corpus/test"], help="Input directories containing stage03 papers")
    parser.add_argument("--output_dir", type=str, default="data/corpus/s2and_prescience", help="Output directory for S2AND feature files")
    parser.add_argument("--batch_size", type=int, default=500, help="Batch size for S2 API calls")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing files")
    args = parser.parse_args()

    # Load stage03 papers from all splits
    all_papers = []
    for input_dir in args.input_dirs:
        records, _ = utils.load_json(os.path.join(input_dir, "all_papers.stage03.json"))
        all_papers.extend(records)
    all_papers = [p for p in all_papers if "authors" in p]
    all_papers_dict = {paper["corpus_id"]: paper for paper in all_papers}

    author_ids = set()
    for p in all_papers:
        if "authors" in p:
            author_ids.update(a["author_id"] for a in p["authors"])
    corpus_ids = list(all_papers_dict.keys())
    author_ids = list(author_ids)

    utils.log(f"Found {len(corpus_ids)} unique corpus IDs in target papers.")
    utils.log(f"Found {len(author_ids)} unique author IDs in target papers.")

    # Batch-fetch paper metadata and SPECTER embeddings
    utils.log("Fetching paper metadata and SPECTER embeddings from S2 API")
    s2_ids = [f"CorpusId:{cid}" for cid in corpus_ids]
    paper_records = utils.s2_batch_lookup(
        s2_ids,
        url=f"{utils.S2_API_BASE}/paper/batch",
        fields=["corpusId", "title", "abstract", "venue", "year", "journal", "embedding"],
        batch_size=args.batch_size,
    )

    paper_metadata = {}
    specter_dict = {}
    for record in paper_records:
        if record is None or record.get("corpusId") is None:
            continue
        corpus_id = str(record["corpusId"])
        paper_metadata[corpus_id] = record
        embedding = record.get("embedding")
        if embedding is not None and embedding.get("vector") is not None:
            specter_dict[corpus_id] = np.array(embedding["vector"])

    utils.log(f"Fetched metadata for {len(paper_metadata)} papers, {len(specter_dict)} with SPECTER embeddings")

    # Fetch ALL references (not just influential) per paper for S2AND
    utils.log("Fetching all references per paper for S2AND")
    ref_records = utils.s2_batch_lookup(
        s2_ids,
        url=f"{utils.S2_API_BASE}/paper/batch",
        fields=["corpusId", "references", "references.corpusId"],
        batch_size=args.batch_size,
    )
    references_dict = {}
    for record in ref_records:
        if record is None or record.get("corpusId") is None:
            continue
        corpus_id = str(record["corpusId"])
        refs = record.get("references") or []
        references_dict[corpus_id] = [
            int(ref["corpusId"]) for ref in refs
            if ref is not None and ref.get("corpusId") is not None
        ]
    utils.log(f"Fetched references for {len(references_dict)} papers")

    # Batch-fetch author details (for affiliations; email not available from public API)
    utils.log("Fetching author details from S2 API")
    author_records = utils.s2_batch_lookup(
        author_ids,
        url=f"{utils.S2_API_BASE}/author/batch",
        fields=["authorId", "name", "affiliations"],
        batch_size=1000,
    )
    author_records_dict = {}
    for record in author_records:
        if record is not None and record.get("authorId") is not None:
            author_records_dict[str(record["authorId"])] = record

    utils.log(f"Fetched details for {len(author_records_dict)} authors")

    # Build papers.json and signatures.json
    papers_json_dict = {}
    signatures_json_dict = {}
    for corpus_id in tqdm(corpus_ids, desc="Creating S2AND features dicts"):
        if corpus_id not in paper_metadata:
            continue
        record = paper_metadata[corpus_id]
        journal_name = (record.get("journal") or {}).get("name")
        papers_json_dict[corpus_id] = {
            "paper_id": int(corpus_id),
            "title": record.get("title") or "",
            "abstract": record.get("abstract") or "",
            "journal_name": journal_name,
            "venue": record.get("venue"),
            "year": record.get("year"),
            "references": references_dict.get(corpus_id, []),
            "authors": [
                {"position": idx, "author_name": a["name"]}
                for idx, a in enumerate(all_papers_dict[corpus_id]["authors"])
            ] if "authors" in all_papers_dict[corpus_id] else []
        }

        for idx, a in enumerate(all_papers_dict[corpus_id]["authors"]):
            signature_id = f"{corpus_id}_{a['author_id']}"
            if a["author_id"] in author_records_dict:
                author_record = author_records_dict[a["author_id"]]
                first, middle, last = split_author_name(author_record.get("name", ""))
                affiliations = author_record.get("affiliations") or []
            else:
                first, middle, last = split_author_name(a.get("name", ""))
                affiliations = []
            block = get_block({"first_name": first, "last_name": last})
            signatures_json_dict[signature_id] = {
                "author_id": int(a["author_id"]),
                "paper_id": int(corpus_id),
                "signature_id": signature_id,
                "author_info": {
                    "given_block": block,
                    "block": block,
                    "position": idx,
                    "first": first,
                    "middle": middle,
                    "last": last,
                    "suffix": None,
                    "affiliations": affiliations,
                    "email": None,
                }
            }

    # Build SPECTER embedding pickle
    embeddings_array = np.array(list(specter_dict.values()))
    corpus_ids_array = np.array(list(specter_dict.keys()))
    specter_tuple = (embeddings_array, corpus_ids_array)

    utils.log(f"Saving {len(papers_json_dict)} papers, {len(signatures_json_dict)} signatures, {len(specter_dict)} embeddings")
    utils.save_json(papers_json_dict, os.path.join(args.output_dir, "papers.json"), overwrite=args.overwrite)
    utils.save_json(signatures_json_dict, os.path.join(args.output_dir, "signatures.json"), overwrite=args.overwrite)
    utils.save_pkl(specter_tuple, os.path.join(args.output_dir, "specter.pickle"), overwrite=args.overwrite, protocol=4)


if __name__ == "__main__":
    main()
