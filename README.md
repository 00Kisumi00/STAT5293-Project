# Improving Faithfulness in RAG

This project studies how retrieval, reranking, and prompting strategies affect answer quality and faithfulness in a simple Retrieval-Augmented Generation (RAG) question answering system.

The system uses a shuffled subset of 2,000 SQuAD validation examples as the document collection. It compares BM25 sparse retrieval, dense embedding retrieval, hybrid retrieval, cross-encoder reranking, and multiple prompting strategies.

## Project Motivation

Large language models can generate fluent answers, but they may also produce information that is not supported by evidence. Retrieval-Augmented Generation (RAG) is a common approach for reducing this problem by retrieving relevant passages and using them as context for answer generation.

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
- **Short-Grounded Prompt**: asks the model to return only the shortest possible answer span.

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

### Oracle Context Check

In addition to normal RAG evaluation, this project includes an oracle-context diagnostic. In this setting, the generator receives the gold context directly instead of relying on retrieved passages.

This helps separate:

- retrieval and ranking limitations,
- prompt construction issues,
- and the generator's answer extraction ability.

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
- `correct_supported`
- `partially_supported`
- `format_mismatch`
- `retrieval_failure`

## Main Results

### Retrieval Results

First-stage retrieval was evaluated on 2,000 questions.

| Method | Recall@5 |
|---|---:|
| BM25 | 0.7900 |
| Dense Retrieval | 0.8685 |
| Hybrid Retrieval | 0.9125 |

Reranked retrieval was evaluated on 1,000 questions.

| Method | Recall@5 |
|---|---:|
| BM25 + Rerank | 0.8810 |
| Dense + Rerank | 0.9370 |
| Hybrid + Rerank | 0.9730 |

The best retrieval configuration was **Hybrid + Rerank**, with a Recall@5 of **0.9730**.

### Answer Generation Results

Answer generation was evaluated on 200 questions across ten RAG configurations.

| Configuration | EM | F1 |
|---|---:|---:|
| Dense + Short Grounded | 0.135 | 0.170 |
| Dense + Grounded | 0.120 | 0.158 |
| Dense + Rerank + Short Grounded | 0.115 | 0.135 |
| Dense + Rerank + Grounded | 0.100 | 0.124 |
| Hybrid + Rerank + Short Grounded | 0.095 | 0.115 |
| Hybrid + Rerank + Grounded | 0.085 | 0.110 |
| Hybrid + Rerank + Evidence | 0.070 | 0.078 |
| BM25 + Standard | 0.020 | 0.052 |
| BM25 + Rerank + Grounded | 0.025 | 0.041 |
| BM25 + Grounded | 0.025 | 0.031 |

The best automatic answer quality came from **Dense + Short Grounded**, with an Exact Match score of **0.135** and an F1 score of **0.170**.

### Oracle Context Results

The oracle-context experiment gives the model the gold context directly.

| Configuration | EM | F1 |
|---|---:|---:|
| Oracle Context + Short Grounded | 0.695 | 0.831 |
| Oracle Context + Grounded | 0.660 | 0.786 |

The oracle-context results are much higher than the normal RAG generation results. This suggests that the generator can extract correct answers when the right context is clearly provided, but full RAG performance is limited by the combined effects of retrieval, ranking, context selection, prompt length, and prompt construction.

### Manual Faithfulness Results

In the manual review sample, the highest average faithfulness score was achieved by **Dense + Rerank + Short Grounded**, with an average score of **0.875**.

The next strongest configurations were:

| Configuration | Manual Faithfulness Score |
|---|---:|
| Dense + Rerank + Short Grounded | 0.875 |
| Dense + Rerank + Grounded | 0.750 |
| Hybrid + Rerank + Short Grounded | 0.750 |
| Hybrid + Rerank + Grounded | 0.625 |
| BM25 + Grounded | 0.500 |
| Hybrid + Rerank + Evidence | 0.375 |
| Dense + Short Grounded | 0.375 |
| BM25 + Rerank + Grounded | 0.250 |
| Dense + Grounded | 0.250 |
| BM25 + Standard | 0.000 |

The most common error type was **over-refusal**, where the model answered "Not enough information" even when the retrieved context often contained enough evidence. This shows that grounded prompting may reduce hallucination, but it can also make the model overly conservative.

## Key Findings

The main findings are:

1. **Hybrid retrieval performed best among first-stage retrieval methods.**
2. **Cross-encoder reranking improved Recall@5 and produced the strongest retrieval configuration.**
3. **Short-grounded prompting improved answer quality by encouraging concise answer spans.**
4. **Strong retrieval did not automatically lead to strong generated answers.**
5. **Oracle-context results show that context selection and prompt construction are major bottlenecks.**
6. **Manual faithfulness review is necessary because EM and F1 do not fully measure evidence support.**

## Repository Structure

```text
rag-faithfulness-project/
├── README.md
├── requirements.txt
├── demo_app.py
├── 5293_Final_Project.ipynb
│
├── results/
│   ├── answer_quality_summary.csv
│   ├── manual_faithfulness_review_labeled.csv
│   ├── oracle_context_results.csv
│   ├── oracle_context_summary.csv
│   ├── rag_generation_results.csv
│   ├── representative_error_cases.csv
│   └── selected_error_cases.csv
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
10. oracle-context evaluation,
11. visualization,
12. manual faithfulness review,
13. error analysis,
14. conclusion and future work.

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
4. displays the gold answer when available,
5. displays the retrieved evidence.

The demo uses:

```text
Hybrid Retrieval + Reciprocal-Rank Candidate Merging + Cross-Encoder Reranking + Grounded Prompting
```

The first run may take several minutes because the pretrained models must be downloaded and the retrieval index must be built.

## Demo Notes

The demo is intended to show the full RAG workflow rather than serve as a production system. It allows users to inspect the generated answer, the gold answer when available, and the retrieved evidence. This is useful for evaluating whether the answer is faithful to the retrieved context.

## Limitations

This project has several limitations:

- The dataset is based on SQuAD, where answers are usually short text spans.
- EM and F1 may penalize answers that are semantically reasonable but do not exactly match the gold answer.
- The generator model, `google/flan-t5-base`, is lightweight but not as capable as larger instruction-tuned models.
- The manual faithfulness review uses a small sample and should be interpreted as qualitative evidence.
- The hybrid retrieval method is simple and could be improved with better score normalization or learned fusion.
- The demo builds the retrieval index at startup, so the first run may be slow.

## Future Work

Future improvements could include:

- using a stronger generator model,
- improving prompt construction to prioritize the most relevant evidence,
- reducing prompt truncation effects,
- expanding manual faithfulness review to more examples,
- using multiple annotators for manual evaluation,
- evaluating on more realistic long-document or open-domain QA datasets,
- improving hybrid retrieval with learned retrieval fusion or score calibration.

## Authors

- Joseph Yeung
- Kevin Ma
- Zenan Luo

