import argparse
import numpy as np
from tqdm import tqdm

import utils


def compute_ngram_overlap(generated_text, reference_text, n=1):
    """
    Compute the n-gram overlap between two texts.
    Lowercase all text, split into words without punctuation, 
    and calculate the overlap of n-grams.
    """
    text1, text2 = generated_text, reference_text

    text1 = text1.lower().split()
    text2 = text2.lower().split()

    # remove punctuation
    text1 = [word.strip('.,!?;:"()[]{}') for word in text1]
    text2 = [word.strip('.,!?;:"()[]{}') for word in text2]

    # remove empty strings
    text1 = [word for word in text1 if word]
    text2 = [word for word in text2 if word]

    ngrams1 = set(tuple(text1[i:i+n]) for i in range(len(text1) - n + 1))
    ngrams2 = set(tuple(text2[i:i+n]) for i in range(len(text2) - n + 1))

    overlap = ngrams1.intersection(ngrams2)
    return {
        "precision": len(overlap) / len(ngrams1) if len(ngrams1) > 0 else 0,
        "recall": len(overlap) / len(ngrams2) if len(ngrams2) > 0 else 0,
    }
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ngram overlap evaluation script")
    parser.add_argument("--input_file", type=str, default="data/task_followup_prediction/test/query_papers_gpt-4o-2024-11-20.json", help="Input file containing generated abstracts")
    parser.add_argument("--reference_file", type=str, default="data/corpus/test/query_papers.json", help="Reference file containing original abstracts")
    parser.add_argument("--max_query_papers", type=int, default=5_000, help="Maximum number of query papers to process")
    args = parser.parse_args()

    synthetic_records = utils.load_json(args.input_file)[0][:args.max_query_papers]
    reference_records = utils.load_json(args.reference_file)[0]
    synthetic_records_dict = {rec["corpus_id"]: rec for rec in synthetic_records}
    reference_records_dict = {rec["corpus_id"]: rec for rec in reference_records}

    onegram_overlaps = []
    twogram_overlaps = []
    for corpus_id in tqdm(synthetic_records_dict.keys(), desc="Computing n-gram overlaps"):
        synthetic_tabstract = synthetic_records_dict[corpus_id]["title"] + " " + synthetic_records_dict[corpus_id]["abstract"]
        reference_abstract = reference_records_dict[corpus_id]["title"] + " " + reference_records_dict[corpus_id]["abstract"]
        onegram_overlap = compute_ngram_overlap(synthetic_tabstract, reference_abstract, n=1)
        twogram_overlap = compute_ngram_overlap(synthetic_tabstract, reference_abstract, n=2)
        onegram_overlaps.append(onegram_overlap)
        twogram_overlaps.append(twogram_overlap)
    
    mean_onegram_overlap = {k: np.mean([overlap[k] for overlap in onegram_overlaps]) for k in onegram_overlaps[0].keys()}
    mean_twogram_overlap = {k: np.mean([overlap[k] for overlap in twogram_overlaps]) for k in twogram_overlaps[0].keys()}
    
    print(f"Mean 1-gram overlap: {mean_onegram_overlap}")
    print(f"Mean 2-gram overlap: {mean_twogram_overlap}")