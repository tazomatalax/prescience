"""LACER scoring for followup work prediction. Scores generated title/abstract pairs against references."""

import os
import re
import argparse
import concurrent.futures as cf

import anthropic
import openai
from tqdm import tqdm

import utils

SYSTEM_PROMPT = "You are an expert reviewer that assigns similarity scores to research paper title-abstract pairs."
SCORE_REGEX = re.compile(r"Score\s*:\s*([-+]?[0-9]+(?:\.\d+)?)", re.IGNORECASE)

# Suffix appended to prompt templates for scoring
SCORING_SUFFIX = """Generated Paper:
Title: {{generation_title}}
Abstract: {{generation_abstract}}

Reference Paper:
Title: {{reference_title}}
Abstract: {{reference_abstract}}"""


def parse_score(response_text):
    """Parse numeric score from model response text."""
    match = SCORE_REGEX.search(response_text)
    if not match:
        raise ValueError(f"Unable to parse score from response: {response_text}")
    return float(match.group(1))


def build_prompt(prompt_template, reference, generation):
    """Build scoring prompt from template and paper data."""
    prompt = prompt_template.replace("{{reference_title}}", reference["title"])
    prompt = prompt.replace("{{reference_abstract}}", reference["abstract"])
    prompt = prompt.replace("{{generation_title}}", generation["title"])
    prompt = prompt.replace("{{generation_abstract}}", generation["abstract"])
    return prompt


def _score_generation(task, prompt_template, reference_lookup, model):
    """Score a single generation against its reference. Used by analysis scripts.

    Args:
        task: dict with corpus_id, title, abstract fields
        prompt_template: full prompt template string with placeholders
        reference_lookup: dict mapping corpus_id to reference dict with title/abstract
        model: OpenAI model name

    Returns:
        dict with original task fields plus lacer_score, lacer_response, lacer_scoring_model
    """
    client = openai.OpenAI()
    scored_record = dict(task)

    reference = reference_lookup.get(task["corpus_id"], {})
    generation = {"title": task.get("title", ""), "abstract": task.get("abstract", "")}
    prompt = build_prompt(prompt_template, reference, generation)

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ]
        )
        content = response.choices[0].message.content.strip()
        score = parse_score(content)
        scored_record["lacer_score"] = score
        scored_record["lacer_response"] = content
        scored_record["lacer_scoring_model"] = model
    except Exception as exc:
        scored_record["lacer_error"] = str(exc)

    return scored_record


def score_generation_openai(record):
    """Score a single generation against its reference using OpenAI."""
    client = record["client"]
    prompt_template = record["prompt_template"]
    all_papers_dict = record["all_papers_dict"]
    model = record["model"]
    del record["client"], record["prompt_template"], record["all_papers_dict"], record["model"]

    scored_record = dict(record)
    reference = all_papers_dict[record["corpus_id"]]
    prompt = build_prompt(prompt_template, reference, record)

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ]
        )
        content = response.choices[0].message.content.strip()
        score = parse_score(content)
        scored_record["lacer_score"] = score
        scored_record["lacer_response"] = content
        scored_record["lacer_scoring_model"] = model
    except Exception as exc:
        scored_record["lacer_error"] = str(exc)

    return scored_record


def score_generation_anthropic(record):
    """Score a single generation against its reference using Anthropic."""
    client = record["client"]
    prompt_template = record["prompt_template"]
    all_papers_dict = record["all_papers_dict"]
    model = record["model"]
    del record["client"], record["prompt_template"], record["all_papers_dict"], record["model"]

    scored_record = dict(record)
    reference = all_papers_dict[record["corpus_id"]]
    prompt = build_prompt(prompt_template, reference, record)

    try:
        response = client.messages.create(
            model=model,
            max_tokens=4096,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": prompt}]
        )
        content = response.content[0].text.strip()
        score = parse_score(content)
        scored_record["lacer_score"] = score
        scored_record["lacer_response"] = content
        scored_record["lacer_scoring_model"] = model
    except Exception as exc:
        scored_record["lacer_error"] = str(exc)

    return scored_record


def main():
    """Score generated title/abstract pairs against references using LACER."""
    parser = argparse.ArgumentParser(description="Score generated title+abstract pairs against references using LACER")
    parser.add_argument("--input_path", required=True, help="Path to JSON file with generated title/abstract pairs")
    parser.add_argument("--prompt_path", default="task_followup_prediction/evaluate/templates/lacer_scoring_prompt_percentile50.110.txt", help="Path to LACER scoring prompt template")
    parser.add_argument("--hf_repo_id", type=str, default="allenai/prescience", help="HuggingFace repo ID for dataset")
    parser.add_argument("--split", type=str, default="test", choices=["train", "test"], help="Dataset split to use")
    parser.add_argument("--output_dir", default="data/task_followup_prediction/test/lacer_scored", help="Directory to save the scored JSON output")
    parser.add_argument("--judge", default="openai", choices=["openai", "anthropic"], help="Which API to use for scoring")
    parser.add_argument("--model", default="gpt-5-2025-08-07", help="Model to use for scoring")
    parser.add_argument("--num_workers", type=int, default=128)
    parser.add_argument("--max_records", type=int, default=None, help="Optional cap on number of generations to score")
    parser.add_argument("--overwrite", action="store_true", help="Allow overwriting the output file if it exists")
    args = parser.parse_args()

    # Load generated title/abstract pairs and reference corpus
    utils.log(f"Loading generations from {args.input_path}")
    generations, metadata = utils.load_json(args.input_path)
    if args.max_records is not None:
        generations = generations[:args.max_records]

    utils.log(f"Loading corpus from HuggingFace repo {args.hf_repo_id} (split={args.split})")
    all_papers, _, _ = utils.load_corpus(hf_repo_id=args.hf_repo_id, split=args.split, embedding_type=None, load_sd2publications=False)
    all_papers_dict = {p["corpus_id"]: p for p in all_papers}

    # Load LACER scoring prompt template
    with open(args.prompt_path, "r", encoding="utf-8") as f:
        prompt_template = f.read().strip()

    # Initialize API client and select scoring function based on judge
    if args.judge == "openai":
        client = openai.OpenAI()
        score_fn = score_generation_openai
    else:
        client = anthropic.Anthropic()
        score_fn = score_generation_anthropic

    # Prepare jobs with all required context for parallel scoring
    utils.log(f"Scoring {len(generations)} generations with {args.num_workers} workers using {args.judge} ({args.model})")
    jobs = []
    for record in generations:
        record["client"] = client
        record["prompt_template"] = prompt_template
        record["all_papers_dict"] = all_papers_dict
        record["model"] = args.model
        jobs.append(record)

    # Score all generations in parallel using thread pool
    with cf.ThreadPoolExecutor(max_workers=args.num_workers) as executor:
        results_iter = executor.map(score_fn, jobs)
        scored_records = list(tqdm(results_iter, total=len(jobs), desc="Scoring generations"))

    error_count = sum(1 for record in scored_records if "lacer_error" in record)
    utils.log(f"Finished scoring {len(scored_records)} records; errors encountered on {error_count} records")

    scores = [r["lacer_score"] for r in scored_records if "lacer_score" in r]
    utils.log(f"Mean LACER score: {sum(scores) / len(scores):.2f} (n={len(scores)})")

    # Save scored records to output file
    input_filename = os.path.basename(args.input_path)
    output_filename = os.path.splitext(input_filename)[0] + ".lacer_scored.json"
    output_path = os.path.join(args.output_dir, output_filename)

    utils.save_json(scored_records, output_path, metadata=utils.update_metadata(metadata, args), overwrite=args.overwrite)
    utils.log(f"Saved scored records to {output_path}")


if __name__ == "__main__":
    main()
