# Improving Faithfulness in RAG

This project studies how retrieval, reranking, and prompting strategies affect answer quality and faithfulness in a simple Retrieval-Augmented Generation (RAG) question answering system.

The system uses a shuffled subset of 2,000 SQuAD validation examples as the document collection. It compares BM25 sparse retrieval, dense embedding retrieval, hybrid retrieval, cross-encoder reranking, and multiple prompting strategies.

## Project Motivation

Large language models can generate fluent answers, but they may also produce information that is not supported by evidence. RAG is a common approach for reducing this problem by retrieving relevant passages and using them as context for answer generation.

However, final answer quality depends on several design choices:

- how passages are retrieved,
- whether retrieved passages are reranked,
- how the prompt instructs the model to use evidence,
- and whether the generator can extract the correct answer from the retrieved context.

This project evaluates these design choices in a controlled RAG pipeline.

## Research Questions

This project focuses on four main questions:

1. Does the retrieval method affect whether the system finds answer-supporting evidence?
2. Does cross-encoder reranking improve retrieval quality?
3. Do grounded prompting strategies improve answer faithfulness?
4. Which combination of retrieval, reranking, and prompting works best in this simple RAG pipeline?

## Methods

The project compares the following components.

### Retrieval Methods

- **BM25 Retrieval**: sparse keyword-based retrieval baseline.
- **Dense Retrieval**: semantic retrieval using `sentence-transformers/all-MiniLM-L6-v2`.
- **Hybrid Retrieval**: combines BM25 and dense retrieval using reciprocal-rank style candidate merging.

### Reranking

- **No Reranking**
- **Cross-Encoder Reranking** using `cross-encoder/ms-marco-MiniLM-L-6-v2`

### Prompting Strategies

- **Standard Prompt**: asks the model to answer using the retrieved context.
- **Grounded Prompt**: asks the model to answer only from the retrieved context and say "Not enough information" if unsupported.
- **Evidence Prompt**: asks the model to answer and identify supporting evidence.

### Generator

The answer generation model is:

```text
google/flan-t5-base
```

This model was selected because it is lightweight enough to run in Google Colab while still supporting instruction-style prompting.

## Dataset

The project uses a shuffled subset of the SQuAD validation set:

```text
2,000 validation examples
```

The contexts from these examples are used as the document collection. Each context is split into overlapping chunks before retrieval.

## Evaluation

The project uses both automatic and manual evaluation.

### Automatic Metrics

- **Recall@5** for retrieval evaluation
- **Exact Match (EM)** for answer quality
- **Token-level F1** for answer quality

### Manual Faithfulness Review

Since EM and F1 do not directly measure whether an answer is supported by evidence, a small manual review was also conducted.

Each sampled answer was labeled as:

| Label | Meaning |
|---:|---|
| 1.0 | Fully supported by retrieved context |
| 0.5 | Partially supported, incomplete, vague, or overly conservative |
| 0.0 | Unsupported, incorrect, or hallucinated |

Error types were also recorded, including:

- `over_refusal`
- `wrong_answer`
- `partially_supported`
- `format_mismatch`
- `retrieval_failure`

## Main Results

### Retrieval Results

First-stage retrieval was evaluated on 2,000 questions.

| Method | Recall@5 |
|---|---:|
| BM25 | 0.5990 |
| Dense Retrieval | 0.6575 |
| Hybrid Retrieval | 0.6885 |

Reranked retrieval was evaluated on 1,000 questions.

| Method | Recall@5 |
|---|---:|
| BM25 + Rerank | 0.8990 |
| Dense + Rerank | 0.9530 |
| Hybrid + Rerank | 0.9790 |

The best retrieval configuration was **Hybrid + Rerank**, with a Recall@5 of **0.9790**.

### Answer Generation Results

Answer generation was evaluated on 200 questions across seven RAG configurations.

| Configuration | EM | F1 |
|---|---:|---:|
| Dense + Grounded | 0.130 | 0.167 |
| Dense + Rerank + Grounded | 0.100 | 0.120 |
| Hybrid + Rerank + Grounded | 0.080 | 0.097 |
| BM25 + Standard | 0.065 | 0.093 |
| Hybrid + Rerank + Evidence | 0.075 | 0.092 |
| BM25 + Grounded | 0.045 | 0.060 |
| BM25 + Rerank + Grounded | 0.015 | 0.023 |

The best automatic answer quality came from **Dense + Grounded**.

## Key Findings

The main findings are:

1. **Hybrid retrieval performed best among first-stage retrievers.**
2. **Cross-encoder reranking substantially improved Recall@5.**
3. **Improved retrieval did not automatically produce better generated answers.**
4. **Dense retrieval with grounded prompting produced the best automatic answer quality.**
5. **Grounded prompting helped reduce unsupported generation but caused over-refusal.**
6. **Manual faithfulness evaluation is necessary because EM and F1 do not fully capture evidence support.**

## Repository Structure

```text
rag-faithfulness-project/
├── README.md
├── requirements.txt
├── demo_app.py
├── 5293_Final_Project.ipynb
│
├── results/
│   ├── retrieval_summary.csv
│   ├── answer_quality_summary.csv
│   ├── rag_generation_results.csv
│   ├── manual_faithfulness_review_labeled_new.csv
│   └── representative_error_cases.csv
│
└── figures/
    ├── retrieval_performance_comparison.png
    ├── answer_quality_comparison.png
    └── manual_faithfulness_comparison.png
```

## Setup

Install dependencies:

```bash
pip install -r requirements.txt
```

The main dependencies are:

```text
numpy
pandas
matplotlib
tqdm
datasets
rank-bm25
sentence-transformers
faiss-cpu
transformers
accelerate
torch
gradio
evaluate
rouge-score
```

## Running the Experiments

Open the notebook:

```text
5293_Final_Project.ipynb
```

The notebook includes:

1. dataset loading,
2. document chunking,
3. BM25 retrieval,
4. dense retrieval,
5. hybrid retrieval,
6. cross-encoder reranking,
7. prompting strategies,
8. answer generation,
9. automatic evaluation,
10. visualization,
11. manual faithfulness review,
12. error analysis,
13. conclusion and future work.

The notebook was designed to run in Google Colab.

## Running the Demo

Run the Gradio demo:

```bash
python demo_app.py
```

The demo launches a web interface where users can enter a question. The system then:

1. retrieves passages using hybrid retrieval,
2. applies cross-encoder reranking,
3. generates a grounded answer,
4. displays the retrieved evidence.

The demo uses:

```text
Hybrid Retrieval + Reciprocal-Rank Candidate Merging + Cross-Encoder Reranking + Grounded Prompting
```

The first run may take several minutes because the pretrained models must be downloaded and the retrieval index must be built.

## Demo Notes

The demo is intended to show the full RAG workflow rather than serve as a production system. It allows users to inspect both the generated answer and the retrieved evidence, which is important for evaluating faithfulness.

## Limitations

This project has several limitations:

- The dataset is based on SQuAD, where answers are usually short text spans.
- EM and F1 may penalize answers that are semantically reasonable but do not exactly match the gold answer.
- The generator model, `google/flan-t5-base`, is lightweight but not as capable as larger instruction-tuned models.
- The manual faithfulness review uses a small sample and should be interpreted as qualitative evidence.
- The hybrid retrieval method is simple and could be improved with better score normalization or learned fusion.

## Future Work

Future improvements could include:

- using a stronger generator model,
- designing prompts that force short answer-span extraction,
- expanding manual faithfulness review to more examples,
- using multiple annotators for manual evaluation,
- evaluating on more realistic long-document or open-domain QA datasets,
- improving hybrid retrieval with learned retrieval fusion or score calibration.

