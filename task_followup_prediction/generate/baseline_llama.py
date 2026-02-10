"""Llama baseline for followup work prediction. Generates title/abstract predictions using local Llama model."""

import os
import argparse
import torch
import transformers
from tqdm import tqdm

import utils
from task_followup_prediction.dataset import get_query_papers, create_messages


def query_llama(pipeline, messages):
    """Query Llama model and parse the response into title, abstract, and reasoning."""
    try:
        response = pipeline(messages, max_new_tokens=4096*4)
        answer = response[0]["generated_text"][-1]["content"].strip()

        reasoning = answer.split("Reasoning: ")[1].split("Title: ")[0].strip()
        title = answer.split("Title: ")[1].split("Abstract: ")[0].strip()
        abstract = answer.split("Abstract: ")[1].strip()
        return {"reasoning": reasoning, "title": title, "abstract": abstract}

    except Exception as e:
        utils.log(f"Error: {e}")
        return {"reasoning": "", "title": "", "abstract": ""}


def main():
    parser = argparse.ArgumentParser(description="Llama baseline for followup work prediction")
    parser.add_argument("--hf_repo_id", type=str, default="allenai/prescience", help="HuggingFace repo ID")
    parser.add_argument("--split", type=str, default="test", choices=["train", "test"], help="Dataset split")
    parser.add_argument("--output_dir", type=str, default="data/task_followup_prediction/test/generations", help="Output directory")
    parser.add_argument("--model", type=str, default="meta-llama/Meta-Llama-3.1-8B-Instruct", help="Model to use")
    parser.add_argument("--max_query_papers", type=int, default=5000, help="Maximum papers to evaluate")
    args = parser.parse_args()

    all_papers, _, _ = utils.load_corpus(hf_repo_id=args.hf_repo_id, split=args.split, embedding_type=None, load_sd2publications=False)
    all_papers_dict = {p["corpus_id"]: p for p in all_papers}
    query_papers = get_query_papers(all_papers, max_papers=args.max_query_papers)

    with open("task_followup_prediction/templates/prediction_system.prompt", "r") as f:
        system_prompt = f.read()

    utils.log(f"Loaded {len(all_papers)} papers, {len(query_papers)} query papers after filtering")

    pipeline = transformers.pipeline("text-generation", model=args.model, tokenizer=args.model, model_kwargs={"torch_dtype": torch.bfloat16}, device_map="auto")

    for idx, record in enumerate(tqdm(query_papers, desc="Generating samples")):
        messages = create_messages(record, system_prompt, all_papers_dict)
        result = query_llama(pipeline, messages)
        record["title"] = result["title"]
        record["abstract"] = result["abstract"]
        record["reasoning"] = result["reasoning"]

        if (idx + 1) % 100 == 0 or (idx + 1) == len(query_papers):
            output_path = os.path.join(args.output_dir, f"generations.{args.model.replace('/', '_')}.json")
            utils.save_json(query_papers, output_path, overwrite=True, metadata={"args": vars(args)})

    utils.log(f"Saved {len(query_papers)} generations to {output_path}")


if __name__ == "__main__":
    main()
