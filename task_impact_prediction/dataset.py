"""Dataset creation utilities for impact prediction task."""

import os
import numpy as np
from tqdm import tqdm

import utils


def load_corpus_impact(hf_repo_id="allenai/prescience", split="test", embeddings_dir=None, embedding_type=None):
    """Load corpus from HuggingFace, build lookup dict, and load embeddings.

    Args:
        hf_repo_id: HuggingFace repo ID (default: "allenai/prescience")
        split: Dataset split ('train' or 'test', default: 'test')
        embeddings_dir: Local directory with embedding files (required)
        embedding_type: Embedding type to load ('gtr', 'specter2', 'grit')

    Returns (all_papers, all_papers_dict, embeddings, metadata).
    """
    all_papers, _, embeddings = utils.load_corpus(
        hf_repo_id=hf_repo_id,
        split=split,
        embeddings_dir=embeddings_dir,
        embedding_type=embedding_type,
        load_sd2publications=False
    )
    all_papers_dict = {p["corpus_id"]: p for p in all_papers}
    metadata = {"source": "huggingface", "repo_id": hf_repo_id, "split": split}
    return all_papers, all_papers_dict, embeddings, metadata


def create_evaluation_instances(all_papers, impact_months):
    """Create evaluation instances for impact prediction.

    Returns list of (date, instance) tuples where instance contains:
      - corpus_id
      - gt_citations (citation_trajectory[impact_months - 1])

    Filters to target papers with len(citation_trajectory) >= impact_months.
    """
    target_papers = utils.filter_by_roles(all_papers, ["target"])
    candidate_papers = [p for p in target_papers if len(p["citation_trajectory"]) >= impact_months]
    candidate_papers_sorted = sorted(candidate_papers, key=lambda p: p["date"])

    instances = []
    for paper in candidate_papers_sorted:
        instances.append((paper["date"], {
            "corpus_id": paper["corpus_id"],
            "gt_citations": paper["citation_trajectory"][impact_months - 1],
        }))

    return instances


def get_embedding_dim(all_embeddings):
    """Get embedding dimension from first available embedding."""
    for emb in all_embeddings.values():
        return emb["key"].reshape(-1).shape[0]
    return 0


def create_features(record, all_papers_dict, all_embeddings, embedding_type,
                    use_author_names, use_author_numbers, use_author_papers,
                    use_prior_work_papers, use_prior_work_numbers, use_followup_work_paper,
                    author_embedding_cache, embed_dim=None):
    """Extract feature vector for a single paper.

    Returns numpy array of features based on enabled flags.
    """
    if embed_dim is None:
        embed_dim = get_embedding_dim(all_embeddings)

    features = []
    authors = record["authors"]

    if use_author_names:
        if authors:
            author_names = [author["name"] for author in authors]
            names_to_embed = [name for name in author_names if name not in author_embedding_cache]

            if names_to_embed:
                unique_names_to_embed = list(dict.fromkeys(names_to_embed))
                embedded_authors = utils.embed_on_gpus(unique_names_to_embed, embedding_type, quiet=True)
                for name, embedding in zip(unique_names_to_embed, embedded_authors):
                    author_embedding_cache[name] = embedding["key"].reshape(-1)

            author_embeddings = [author_embedding_cache[name] for name in author_names]

            if len(author_embeddings) == 1:
                for _ in range(4):
                    features.extend(author_embeddings[0].tolist())
            else:
                features.extend(author_embeddings[0].tolist())
                features.extend(author_embeddings[1].tolist())
                features.extend(author_embeddings[-2].tolist())
                features.extend(author_embeddings[-1].tolist())

            features.extend(np.mean(author_embeddings, axis=0).tolist())
        else:
            features.extend([0.0] * (5 * embed_dim))

    if use_author_numbers:
        if authors:
            author_hindices = [author["h_index"] for author in authors]
            author_prior_papers = [author["num_papers"] for author in authors]
            author_avg_citations = [
                author["num_citations"] / author["num_papers"] if author["num_papers"] > 0 else 0
                for author in authors
            ]

            if len(authors) == 1:
                for _ in range(4):
                    features.extend([author_hindices[0], author_prior_papers[0], author_avg_citations[0]])
            else:
                features.extend([author_hindices[0], author_prior_papers[0], author_avg_citations[0]])
                features.extend([author_hindices[1], author_prior_papers[1], author_avg_citations[1]])
                features.extend([author_hindices[-2], author_prior_papers[-2], author_avg_citations[-2]])
                features.extend([author_hindices[-1], author_prior_papers[-1], author_avg_citations[-1]])

            features.append(float(np.mean(author_hindices)))
            features.append(float(np.mean(author_prior_papers)))
            features.append(float(np.mean(author_avg_citations)))
            features.append(float(np.max(author_hindices)))
            features.append(float(np.max(author_prior_papers)))
            features.append(len(authors))
            features.append(sum(1 for author in authors if author["num_papers"] == 0))
        else:
            features.extend([0.0] * 19)

    if use_author_papers:
        author_mean_embeddings = []
        if authors:
            for author in authors:
                pub_history = author.get("publication_history", [])
                if not pub_history:
                    continue
                pub_embeddings = [
                    all_embeddings[corpus_id]["key"].reshape(-1)
                    for corpus_id in pub_history
                    if corpus_id in all_embeddings
                ]
                if pub_embeddings:
                    author_mean_embeddings.append(np.mean(pub_embeddings, axis=0))

        if author_mean_embeddings:
            if len(author_mean_embeddings) == 1:
                for _ in range(4):
                    features.extend(author_mean_embeddings[0].tolist())
            else:
                features.extend(author_mean_embeddings[0].tolist())
                features.extend(author_mean_embeddings[1].tolist())
                features.extend(author_mean_embeddings[-2].tolist())
                features.extend(author_mean_embeddings[-1].tolist())
            features.extend(np.mean(author_mean_embeddings, axis=0).tolist())
        else:
            features.extend([0.0] * (5 * embed_dim))

    if use_prior_work_papers or use_prior_work_numbers:
        key_references = record["key_references"]
        key_ref_papers = []
        if key_references:
            key_ref_ids = [ref["corpus_id"] for ref in key_references]
            key_ref_papers = [all_papers_dict[cid] for cid in key_ref_ids if cid in all_papers_dict]

    if use_prior_work_papers:
        features.append(len(key_references))

        ref_embeddings = []
        if key_ref_papers:
            missing_refs = [p for p in key_ref_papers if p["corpus_id"] not in all_embeddings]

            if missing_refs:
                ref_strings = [utils.get_title_abstract_string(p) for p in missing_refs]
                embedded_refs = utils.embed_on_gpus_parallel(ref_strings, embedding_type, quiet=True)
                for paper, embedding in zip(missing_refs, embedded_refs):
                    all_embeddings[paper["corpus_id"]] = embedding

            ref_embeddings = [
                all_embeddings[p["corpus_id"]]["key"].reshape(-1)
                for p in key_ref_papers
                if p["corpus_id"] in all_embeddings
            ]

        if ref_embeddings:
            features.extend(np.mean(ref_embeddings, axis=0).tolist())
        else:
            features.extend([0.0] * embed_dim)

    if use_prior_work_numbers:
        ref_citations = [p.get("citation_count", 0) for p in key_ref_papers] if key_ref_papers else []
        if ref_citations:
            features.append(float(np.mean(ref_citations)))
            features.append(float(np.max(ref_citations)))
        else:
            features.extend([0.0, 0.0])

    if use_followup_work_paper:
        corpus_id = record["corpus_id"]
        if corpus_id in all_embeddings:
            paper_embedding = all_embeddings[corpus_id]["key"].reshape(-1)
        else:
            embedded_paper = utils.embed_on_gpus_parallel(
                [utils.get_title_abstract_string(record)], embedding_type, quiet=True
            )[0]
            all_embeddings[corpus_id] = embedded_paper
            paper_embedding = embedded_paper["key"].reshape(-1)
        features.extend(paper_embedding.tolist())

    return np.array(features, dtype=np.float32)


def precompute_author_name_embeddings(papers, embedding_type, author_embedding_cache):
    """Batch embed all unique author names across papers."""
    all_names = set()
    for paper in papers:
        for author in paper.get("authors", []):
            all_names.add(author["name"])

    uncached_names = [n for n in all_names if n not in author_embedding_cache]
    if not uncached_names:
        return

    utils.log(f"Batch embedding {len(uncached_names)} unique author names")
    embedded = utils.embed_on_gpus(uncached_names, embedding_type)
    for name, emb in zip(uncached_names, embedded):
        author_embedding_cache[name] = emb["key"].reshape(-1)


def build_feature_matrix(papers, all_papers_dict, embeddings, embedding_type,
                         use_author_names, use_author_numbers, use_author_papers,
                         use_prior_work_papers, use_prior_work_numbers, use_followup_work_paper,
                         author_embedding_cache, desc="Building features"):
    """Build feature matrix for a list of papers.

    Returns (X, corpus_ids) where X is a numpy array of shape (n_papers, n_features).
    """
    if use_author_names:
        precompute_author_name_embeddings(papers, embedding_type, author_embedding_cache)

    embed_dim = get_embedding_dim(embeddings)
    X = []
    corpus_ids = []

    for paper in tqdm(papers, desc=desc):
        features = create_features(
            paper, all_papers_dict, embeddings, embedding_type,
            use_author_names, use_author_numbers, use_author_papers,
            use_prior_work_papers, use_prior_work_numbers, use_followup_work_paper,
            author_embedding_cache, embed_dim=embed_dim
        )
        if len(features) > 0:
            X.append(features)
            corpus_ids.append(paper["corpus_id"])

    return np.array(X, dtype=np.float32), corpus_ids


def get_papers_for_instances(instances, all_papers_dict, require_key_references=False):
    """Get paper records for evaluation instances.

    If require_key_references=True, filters to papers with non-empty key_references.
    """
    papers = []
    for _, instance in instances:
        corpus_id = instance["corpus_id"]
        if corpus_id in all_papers_dict:
            paper = all_papers_dict[corpus_id]
            if require_key_references and len(paper["key_references"]) == 0:
                continue
            papers.append(paper)
    return papers
