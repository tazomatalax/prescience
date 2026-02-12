"""Add key (influential) references to target papers."""
import os
import argparse

import utils


def fetch_key_references(all_papers_dict, args):
    """Fetch key (influential) references for target papers from S2 batch API."""
    target_corpus_ids = [cid for cid, p in all_papers_dict.items() if "target" in p["roles"]]
    utils.log(f"Fetching key references for {len(target_corpus_ids)} target papers")

    s2_ids = [f"CorpusId:{cid}" for cid in target_corpus_ids]
    records = utils.s2_batch_lookup(
        s2_ids,
        url=f"{utils.S2_API_BASE}/paper/batch",
        fields=["corpusId", "references", "references.isInfluential", "references.corpusId",
                "references.externalIds", "references.title", "references.abstract", "references.publicationDate"],
        batch_size=args.batch_size,
    )

    key_ref_dict = {}
    key_ref_papers_dict = {}
    for record in records:
        if record is None or record.get("corpusId") is None:
            continue
        target_id = str(record["corpusId"])
        if target_id not in all_papers_dict:
            continue
        paper_date = all_papers_dict[target_id]["date"]
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
                    "categories": [],  # populated in Stage 6 from arXiv snapshot
                    "roles": [],
                }

        if influential_refs:
            key_ref_dict[target_id] = list(set(influential_refs))

    total_refs = sum(len(refs) for refs in key_ref_dict.values())
    utils.log(f"Found {total_refs} key references across {len(key_ref_dict)} target papers ({len(target_corpus_ids) - len(key_ref_dict)} targets have no key references)")
    return key_ref_dict, key_ref_papers_dict


def attach_and_insert_key_references(all_papers_dict, key_ref_dict, key_ref_papers_dict, args):
    """Attach key references to target papers, filter by max count, and insert papers into corpus."""
    key_ref_dict = {k: v for k, v in key_ref_dict.items() if len(v) <= args.max_key_references}
    all_papers_dict = {
        corpus_id: paper for corpus_id, paper in all_papers_dict.items()
        if "target" not in paper["roles"] or corpus_id in key_ref_dict
    }
    utils.log(f"Retained {len(key_ref_dict)} target papers with at most {args.max_key_references} key references")

    corpus_size_before = len(all_papers_dict)
    for corpus_id, ref_ids in key_ref_dict.items():
        all_papers_dict[corpus_id]["key_references"] = [{"corpus_id": ref_id} for ref_id in ref_ids]
        for ref_id in ref_ids:
            if ref_id not in all_papers_dict:
                all_papers_dict[ref_id] = key_ref_papers_dict[ref_id]
            if "target.key_reference" not in all_papers_dict[ref_id]["roles"]:
                all_papers_dict[ref_id]["roles"].append("target.key_reference")

    utils.log(f"Inserted {len(all_papers_dict) - corpus_size_before} new key reference papers (corpus now contains {len(all_papers_dict)} papers)")
    return all_papers_dict


def main():
    """Add key references to target papers."""
    parser = argparse.ArgumentParser(description="Add key references to target papers")
    parser.add_argument("--input_dir", type=str, default="data/corpus/test", help="Input directory containing stage01 papers")
    parser.add_argument("--output_dir", type=str, default="data/corpus/test", help="Output directory")
    parser.add_argument("--max_key_references", type=int, default=10, help="Max key references per target paper")
    parser.add_argument("--batch_size", type=int, default=500, help="Batch size for S2 API calls")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing output")
    args = parser.parse_args()

    input_path = os.path.join(args.input_dir, "all_papers.stage01.json")
    output_path = os.path.join(args.output_dir, "all_papers.stage02.json")

    utils.log(f"Loading {input_path}")
    all_papers, metadata = utils.load_json(input_path)
    all_papers_dict = {p["corpus_id"]: p for p in all_papers}

    key_ref_dict, key_ref_papers_dict = fetch_key_references(all_papers_dict, args)
    all_papers_dict = attach_and_insert_key_references(all_papers_dict, key_ref_dict, key_ref_papers_dict, args)

    all_papers_dict = utils.remove_unreachable_papers(all_papers_dict)
    utils.log(f"Corpus contains {len(all_papers_dict)} papers after removing unreachable papers")

    all_papers = list(all_papers_dict.values())
    utils.save_json(all_papers, output_path, utils.update_metadata(metadata, args), overwrite=args.overwrite)
    utils.log(f"Saved {len(all_papers)} papers to {output_path}")


if __name__ == "__main__":
    main()
