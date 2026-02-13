<div align="center">

# PreScience

**A Benchmark for Forecasting Scientific Contributions**

[Paper](https://arxiv.org/abs/TODO) | [Dataset](https://huggingface.co/datasets/allenai/prescience)

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

</div>

## Overview

Can AI systems trained on the scientific record up to a fixed point in time forecast the scientific advances that follow? PreScience decomposes the research process into four interdependent generative tasks and evaluates models on a curated dataset of 98,000 AI-related arXiv papers (Oct 2023 -- Oct 2025) with disambiguated author identities, temporally aligned scholarly metadata, and a structured graph of 502,000 total papers. We decompose the prediction of a single scientific advance into the following four interdependent prediction problems:

<img width="721" height="326" alt="image" src="https://github.com/user-attachments/assets/1698f74b-fd49-4312-a39c-5429b362a3ed" />


| Task | Description | Metrics |
|------|-------------|---------|
| Collaborator Prediction | Predict future co-authors given research context | Precision, Recall, F1, nDCG |
| Prior Work Selection | Identify which papers will be cited as key references | Precision, Recall, F1, nDCG |
| Contribution Generation | Generate the title and abstract of a future paper | LACERScore, BERTScore, ROUGE-L |
| Impact Prediction | Forecast 12-month cumulative citation count | MAE, R², Spearman |

## Setup

```bash
git clone https://github.com/allenai/prescience.git
cd prescience
pip install -r requirements.txt
```

**Prerequisites:**
- Python 3.8+
- The dataset is hosted on [HuggingFace](https://huggingface.co/datasets/allenai/prescience) and downloaded automatically on first use
- Embedding-based task baselines and `multiturn/simulate.py` require precomputed embeddings (see [Computing Embeddings](#computing-embeddings))
- LLM-based baselines (contribution generation, LACERScore) require an OpenAI or Anthropic API key
- SPECTER2 and Qwen3 have dependency conflicts with the main environment; use `requirements_specter2.txt` and `requirements_qwen3.txt` in separate environments


## Codebase Structure

```
prescience/
├── task_coauthor_prediction/     # Collaborator prediction baselines, evaluation and analyses
├── task_priorwork_prediction/    # Prior work selection baselines, evaluation and analyses
├── task_followup_prediction/
│   ├── generate/                 # Contribution generation baselines (GPT, Claude, OLMo, LLaMA, Qwen)
│   └── evaluate/                 # LACERScore, BERTScore, ROUGE-L evaluation
│   └── analysis/                 
├── task_impact_prediction/       # Impact prediction baselines, evaluation and analyses
├── multiturn/
│   ├── simulate.py               # End-to-end multi-turn simulation
│   └── analysis/                 # Simulation analysis scripts
├── dataset/
│   ├── corpus/                   # 7-stage corpus creation pipeline
│   ├── embeddings/               # Embedding computation
│   └── s2and_prep/               # S2AND author disambiguation features
├── utils.py                      # Shared utilities (I/O, embeddings, FAISS, S2 API)
└── requirements.txt
```


## Running Baselines

### Collaborator Prediction

```bash
# Predict
python3 -m task_coauthor_prediction.baseline_frequency --split test

# Evaluate
python3 -m task_coauthor_prediction.evaluate --predictions_path data/task_coauthor_prediction/test/predictions/predictions.frequency.one_shot.first.json
```

### Prior Work Selection

```bash
# Predict
python3 -m task_priorwork_prediction.baseline_frequency --split test

# Evaluate
python3 -m task_priorwork_prediction.evaluate --predictions_path data/task_priorwork_prediction/test/predictions/predictions.frequency.json
```

### Contribution Generation

```bash
# Generate (requires OPENAI_API_KEY)
python3 -m task_followup_prediction.generate.baseline_gpt_parallel --split test --model gpt-4o-2024-11-20

# Evaluate with LACERScore (requires OPENAI_API_KEY)
python3 -m task_followup_prediction.evaluate.lacer_score --input_path data/task_followup_prediction/test/generations/generations.gpt-4o-2024-11-20.json
```

### Impact Prediction

```bash
# Predict (requires precomputed embeddings for both train and test splits)
python3 -m task_impact_prediction.baseline_embedding_ridge --split test --train_embeddings_dir data/corpus/train --test_embeddings_dir data/corpus/test --embedding_type grit

# Evaluate
python3 -m task_impact_prediction.evaluate --predictions_path data/task_impact_prediction/test/predictions/predictions.ridge_grit.json
```


## Computing Embeddings

Embedding-based baselines require precomputed embeddings. Compute them for each split you need:

```bash
python3 -m dataset.embeddings.compute_paper_embeddings --split test --embedding_type grit --output_dir data/corpus/test
python3 -m dataset.embeddings.compute_paper_embeddings --split train --embedding_type grit --output_dir data/corpus/train
```

| Type | Model | Dimension | 
|------|-------|-----------|
| `gtr` | sentence-transformers/gtr-t5-large | 768 |
| `specter2` | allenai/specter2_base | 768 |
| `grit` | GritLM/GritLM-7B | 4096 |

SPECTER2 embeddings require a separate environment; install with `pip install -r requirements_specter2.txt`.


## Multi-turn Simulation

The simulation composes all four tasks to generate a synthetic corpus day-by-day over a specified time horizon:

```bash
python3 -m multiturn.simulate --calibration_split train --embeddings_dir data/corpus/train --author_embedding_type grit --paper_embedding_type grit --output_dir data/multiturn/simulated_grit --depth 365
```

Analyze the resulting synthetic corpus:

```bash
python3 -m multiturn.analysis.plot_bootstrapped_diversity --synthetic_dir simulated_grit
python3 -m multiturn.analysis.plot_neologism_analysis --synthetic_dir simulated_grit
```


## Building the Dataset from Scratch

Most users should use the [HuggingFace dataset](https://huggingface.co/datasets/allenai/prescience) directly. The dataset we released was built using internal versions of the Semantic Scholar API for efficiency; the scripts below use the equivalent public APIs, so results may vary slightly.

To rebuild from scratch, you need:
- **S2 API key**: Get one at https://www.semanticscholar.org/product/api#api-key and set `export S2_API_KEY=<your_key>`
- **arXiv snapshot**: Download from [Kaggle](https://www.kaggle.com/datasets/Cornell-University/arxiv) to `data/arxiv_snapshot/arxiv-metadata-oai-snapshot.json`
- **S2AND**: For author disambiguation (Stage 4), follow [S2AND setup](https://github.com/allenai/S2AND)

<details>
<summary><b>Full pipeline commands</b></summary>

Run each stage for both train and test splits. Commands below show the train split; for test, replace `2023-10-01`/`2024-10-01` with `2024-10-01`/`2025-10-01` and `train` with `test`.

```bash
# Stage 1: Download target papers from arXiv snapshot + S2 API
python3 -m dataset.corpus.download_target_papers --start_date 2023-10-01 --end_date 2024-10-01 --output_dir data/corpus/train

# Stage 2: Add key (highly influential) references
python3 -m dataset.corpus.add_key_references --input_dir data/corpus/train --output_dir data/corpus/train

# Stage 3: Add author rosters and publication histories
python3 -m dataset.corpus.add_authors --input_dir data/corpus/train --output_dir data/corpus/train

# Stage 4a: Prepare S2AND input features
python3 -m dataset.s2and_prep.download_s2and_features --input_dirs data/corpus/train data/corpus/test --output_dir data/corpus/s2and_prescience

# Stage 4b: Run S2AND clustering (see https://github.com/allenai/S2AND)

# Stage 4c: Merge disambiguated author identities into corpus
python3 -m dataset.corpus.merge_authors_in_corpus --input_dir data/corpus/train --output_dir data/corpus/train --s2and_data_dir data/corpus/s2and_prescience

# Stage 5: Add citation metadata (citation counts, h-index, trajectories)
python3 -m dataset.corpus.add_citation_metadata --input_dir data/corpus/train --output_dir data/corpus/train

# Stage 6: Replace titles/abstracts with official arXiv versions
python3 -m dataset.corpus.replace_title_abstracts_using_snapshot --input_dir data/corpus/train --output_dir data/corpus/train

# Stage 7: Compute embeddings
python3 -m dataset.embeddings.compute_paper_embeddings --split train --embedding_type gtr --output_dir data/corpus/train
python3 -m dataset.embeddings.compute_paper_embeddings --split train --embedding_type specter2 --output_dir data/corpus/train
python3 -m dataset.embeddings.compute_paper_embeddings --split train --embedding_type grit --output_dir data/corpus/train
```

</details>


## Citation

```bibtex
@article{prescience2026,
  title={PreScience: A Benchmark for Forecasting Scientific Contributions},
  author={Anirudh Ajith and Amanpreet Singh and Jay DeYoung and Nadav Kunievsky and Austin C. Kozlowski and Oyvind Tafjord and James Evans and Daniel S. Weld and Tom Hope and Doug Downey},
  journal={arXiv preprint arXiv:TODO},
  year={2026}
}
```


## License

Apache 2.0 -- see [LICENSE](LICENSE) for details.

For questions or issues, please open an issue on [GitHub](https://github.com/allenai/prescience/issues).
