"""Score annotator generations against ground truths using the LACER prompt."""

import argparse
import concurrent.futures as cf
import json
import os
import threading

import anthropic
import openai
from tqdm import tqdm

import utils
from task_followup_prediction.evaluate.lacer_score import SYSTEM_PROMPT, parse_score, build_prompt

SCORING_SUFFIX = (
    "Before providing the score, write about your reasoning about the similarity or dissimilarity. "
    "Immediately after that reasoning line, provide the similarity score.\n\n"
    "Now please score the following pair of title-abstracts for similarity. \n\n"
    "Reference:\n{{reference_title}}\n{{reference_abstract}}\n\n"
    "Generation:\n{{generation_title}}\n{{generation_abstract}}\n\n"
    "Use the following output format:\nReasoning: ...\nScore: ..."
)
_thread_local = threading.local()


def get_openai_client():
    """Get thread-local OpenAI client."""
    client = getattr(_thread_local, "openai_client", None)
    if client is None:
        client = openai.Client()
        _thread_local.openai_client = client
    return client


def get_anthropic_client():
    """Get thread-local Anthropic client."""
    client = getattr(_thread_local, "anthropic_client", None)
    if client is None:
        client = anthropic.Anthropic()
        _thread_local.anthropic_client = client
    return client


def split_title_abstract(text):
    """Split 'Title\\nAbstract' text into dict with title and abstract keys."""
    if not text:
        return {"title": "", "abstract": ""}
    stripped = text.strip()
    if not stripped:
        return {"title": "", "abstract": ""}
    parts = stripped.split("\n", 1)
    if len(parts) == 1:
        return {"title": parts[0].strip(), "abstract": ""}
    return {"title": parts[0].strip(), "abstract": parts[1].strip()}


def score_pair_openai(prompt_template, model, task):
    """Score a generation against reference using OpenAI."""
    record_idx, generation_id, reference, generation = task
    prompt = build_prompt(prompt_template, reference, generation)
    client = get_openai_client()

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
                {"role": "user", "content": [{"type": "text", "text": prompt}]},
            ],
        )
        message_content = response.choices[0].message.content
        if isinstance(message_content, list):
            content = "".join(part.get("text", "") for part in message_content).strip()
        else:
            content = str(message_content).strip()
        score = parse_score(content)
        result = {"score": score, "response": content, "model": model}
    except Exception as exc:
        result = {"error": str(exc), "model": model}

    return record_idx, generation_id, result


def score_pair_anthropic(prompt_template, model, task):
    """Score a generation against reference using Anthropic."""
    record_idx, generation_id, reference, generation = task
    prompt = build_prompt(prompt_template, reference, generation)
    client = get_anthropic_client()

    try:
        response = client.messages.create(
            model=model,
            max_tokens=4096,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": prompt}],
        )
        content = response.content[0].text.strip()
        score = parse_score(content)
        result = {"score": score, "response": content, "model": model}
    except Exception as exc:
        result = {"error": str(exc), "model": model}

    return record_idx, generation_id, result


def prepare_tasks(records):
    """Prepare scoring tasks from annotator records."""
    tasks = []
    for record_idx, record in enumerate(records):
        reference = split_title_abstract(record.get("ground_truth"))
        generations = record.get("generations", {}) or {}
        for generation_id, generation_text in generations.items():
            if not generation_text:
                continue
            generation = split_title_abstract(generation_text)
            tasks.append((record_idx, generation_id, reference, generation))
    return tasks


def attach_scores(records, results):
    """Attach scoring results to records."""
    for record_idx, generation_id, score_info in results:
        record = records[record_idx]
        if "lacer_scores" not in record:
            record["lacer_scores"] = {}
        record["lacer_scores"][generation_id] = score_info
    return records


def main():
    parser = argparse.ArgumentParser(description="Score annotator generations against ground truths using a LACER prompt")
    parser.add_argument("--input_path", default="data/task_followup_prediction/metrics_analysis/annotator_rankings_rows_2_6.json", help="Path to JSON file with annotator generations")
    parser.add_argument("--prompt_path", default="task_followup_prediction/evaluate/templates/lacer_scoring_prompt_percentile50.110.txt", help="Path to the prompt template used for scoring")
    parser.add_argument("--output_dir", default="data/task_followup_prediction/metrics_analysis", help="Directory to write scored JSON output")
    parser.add_argument("--judge", default="openai", choices=["openai", "anthropic"], help="Which API to use for scoring")
    parser.add_argument("--num_workers", type=int, default=16, help="Number of parallel workers")
    parser.add_argument("--overwrite", action="store_true", help="Allow overwriting the output file if it exists")
    args = parser.parse_args()

    # Determine output filename based on judge
    input_basename = os.path.basename(args.input_path)
    input_name = os.path.splitext(input_basename)[0]
    judge_suffix = "gpt5" if args.judge == "openai" else "opus"
    output_filename = f"{input_name}.{judge_suffix}_scored.percentile50.110.json"
    output_path = os.path.join(args.output_dir, output_filename)

    if os.path.exists(output_path) and not args.overwrite:
        raise FileExistsError(f"Output file already exists: {output_path}. Use --overwrite to replace it.")

    records, _ = utils.load_json(args.input_path)
    tasks = prepare_tasks(records)

    with open(args.prompt_path, "r", encoding="utf-8") as f:
        prompt_prefix = f.read().strip()
    prompt_template = prompt_prefix + "\n\n" + SCORING_SUFFIX

    score_fn, model = (score_pair_openai, "gpt-5-2025-08-07") if args.judge == "openai" else (score_pair_anthropic, "claude-opus-4-5-20251101")

    results = [None] * len(tasks)
    with cf.ThreadPoolExecutor(max_workers=args.num_workers) as executor:
        futures = {executor.submit(score_fn, prompt_template, model, task): idx for idx, task in enumerate(tasks)}
        for future in tqdm(cf.as_completed(futures), total=len(futures), desc="Scoring generations"):
            idx = futures[future]
            results[idx] = future.result()

    records = attach_scores(records, results)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2)


if __name__ == "__main__":
    main()
