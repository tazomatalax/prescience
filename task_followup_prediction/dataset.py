"""Dataset creation utilities for followup work prediction."""

import random
from tqdm import tqdm


def get_query_papers(all_papers, max_papers=None, seed=42):
    """Filter papers to target papers with non-empty key_references, optionally subsampling."""
    random.seed(seed)
    query_papers = [p for p in all_papers if "target" in p["roles"] and len(p["key_references"]) > 0]
    if max_papers is not None and len(query_papers) > max_papers:
        query_papers = random.sample(query_papers, max_papers)
    return query_papers


def format_background_papers(papers):
    """Format a list of papers into the background papers prompt format."""
    prompt = ""
    for i, p in enumerate(papers):
        prompt += f"Background Paper {i+1}:\nTitle: {p['title']}\nAbstract: {p['abstract']}\n\n"
    return prompt


def strip_reasoning_prefix(response):
    """Strip 'Reasoning: ...\n' prefix from a response, keeping only Title and Abstract."""
    if response.startswith("Reasoning:"):
        title_idx = response.find("\nTitle:")
        if title_idx != -1:
            return response[title_idx + 1:]
    return response


def create_messages(record, system_prompt, all_papers_dict):
    """Create chat messages from a target paper's key references."""
    papers = [all_papers_dict[r["corpus_id"]] for r in record["key_references"]]
    user_prompt = format_background_papers(papers)
    return [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]


def create_messages_local(record, system_prompt, all_papers_dict, fewshot_examples, reasoning_trace="generic"):
    """Create chat messages with multi-turn few-shot examples for local models."""
    messages = [{"role": "system", "content": system_prompt}]
    if reasoning_trace == "generic":
        instruction = "\n\nPredict a followup paper that builds on these background papers. Output your response in the format: Reasoning: ... Title: ... Abstract: ..."
    else:
        instruction = "\n\nPredict a followup paper that builds on these background papers. Output your response in the format: Title: ... Abstract: ..."
    for example in fewshot_examples:
        example_response = example["response"] if reasoning_trace == "generic" else strip_reasoning_prefix(example["response"])
        messages.append({"role": "user", "content": format_background_papers(example["background_papers"]) + instruction})
        messages.append({"role": "assistant", "content": example_response})
    papers = [all_papers_dict[r["corpus_id"]] for r in record["key_references"]]
    messages.append({"role": "user", "content": format_background_papers(papers) + instruction})
    return messages


def create_evaluation_instances(all_papers, all_papers_dict):
    """Create chronologically-sorted evaluation instances for followup prediction."""
    candidate_papers = [p for p in all_papers if "target" in p["roles"] and len(p["key_references"]) > 0]
    candidate_papers_sorted = sorted(candidate_papers, key=lambda p: p["date"])

    instances = []
    for paper in tqdm(candidate_papers_sorted, desc="Creating evaluation instances"):
        instances.append((paper["date"], {
            "corpus_id": paper["corpus_id"],
            "key_reference_ids": [ref["corpus_id"] for ref in paper["key_references"]],
            "gt_title": paper["title"],
            "gt_abstract": paper["abstract"],
        }))

    return instances


def create_train_val_split(instances, val_ratio=0.15, seed=42):
    """Split instances into train and validation sets by date (validation = later papers)."""
    random.seed(seed)
    sorted_instances = sorted(instances, key=lambda x: x[0])
    split_idx = int(len(sorted_instances) * (1 - val_ratio))
    return sorted_instances[:split_idx], sorted_instances[split_idx:]


def format_instance_for_training(instance, all_papers_dict, system_prompt, fewshot_examples, reasoning_trace="generic"):
    """Convert an instance to chat message format for instruction tuning with few-shot examples."""
    _, inst = instance

    if reasoning_trace == "generic":
        instruction = "\n\nPredict a followup paper that builds on these background papers. Output your response in the format: Reasoning: ... Title: ... Abstract: ..."
        target_text = f"Reasoning: Based on the background papers, I will predict a followup paper that synthesizes and extends their contributions.\nTitle: {inst['gt_title']}\nAbstract: {inst['gt_abstract']}"
    else:
        instruction = "\n\nPredict a followup paper that builds on these background papers. Output your response in the format: Title: ... Abstract: ..."
        target_text = f"Title: {inst['gt_title']}\nAbstract: {inst['gt_abstract']}"

    messages = [{"role": "system", "content": system_prompt}]
    for example in fewshot_examples:
        example_response = example["response"] if reasoning_trace == "generic" else strip_reasoning_prefix(example["response"])
        messages.append({"role": "user", "content": format_background_papers(example["background_papers"]) + instruction})
        messages.append({"role": "assistant", "content": example_response})

    papers = [all_papers_dict[cid] for cid in inst["key_reference_ids"]]
    user_prompt = format_background_papers(papers) + instruction
    messages.append({"role": "user", "content": user_prompt})
    messages.append({"role": "assistant", "content": target_text})

    return {"messages": messages, "corpus_id": inst["corpus_id"]}
