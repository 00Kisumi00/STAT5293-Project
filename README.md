# Improving Faithfulness in Retrieval-Augmented Generation

This project studies how retrieval, reranking, and prompting strategies affect answer quality and faithfulness in a Retrieval-Augmented Generation (RAG) question answering system.

The goal is not to train a new model or build a production chatbot. Instead, the project is a controlled comparison of RAG pipeline choices: how evidence is retrieved, how candidate passages are reranked, how prompts constrain generation, and how these choices affect both automatic answer quality and evidence-grounded faithfulness.

## Project Summary

We build a simple RAG question answering pipeline using a shuffled subset of 2,000 SQuAD validation examples.

The system compares:

- BM25 sparse retrieval
- Dense retrieval with `sentence-transformers/all-MiniLM-L6-v2`
- Hybrid retrieval using reciprocal-rank style candidate merging
- Cross-encoder reranking with `cross-encoder/ms-marco-MiniLM-L-6-v2`
- Multiple prompting strategies for `google/flan-t5-base`

The evaluation separates three related but different dimensions:

1. **Evidence access**: whether the retriever finds answer-supporting context.
2. **Answer quality**: whether the generated answer matches the gold answer.
3. **Faithfulness**: whether the generated answer is supported by the retrieved evidence.

## Motivation

RAG is commonly used to reduce hallucination by grounding generation in retrieved passages. However, using RAG does not automatically make a model faithful.

A RAG system can fail because:

- the retriever does not find the right evidence,
- the relevant passage is retrieved but ranked poorly,
- the prompt contains too much noisy or truncated context,
- the generator fails to extract the correct answer span,
- or the grounded prompt makes the model too conservative and causes over-refusal.

This project investigates these failure modes through quantitative metrics, oracle-context evaluation, manual faithfulness review, and a small interactive inspection demo.

## Research Questions

1. Which retrieval method finds answer-supporting evidence most reliably?
2. Does cross-encoder reranking improve top-5 evidence access?
3. Which prompting strategy gives the best answer quality under SQuAD-style evaluation?
4. Why can strong retrieval fail to produce strong generated answers?
5. How do automatic metrics and manual faithfulness review differ?

## Dataset

The project uses a shuffled subset of the SQuAD validation set:

```text
2,000 validation examples
```

Each example contains:

- `question`
- `context`
- `answer`

The contexts are treated as the document collection. Each context is split into overlapping word chunks before retrieval:

```text
chunk size = 120 words
overlap = 30 words
```

SQuAD is useful for this project because it provides gold answers and supporting contexts. However, it is also a limitation because many answers are short extractive spans, so metrics such as Exact Match and F1 favor concise span-style outputs.

## Methodology

### Retrieval

We compare three retrieval methods:

| Method | Description |
|---|---|
| BM25 | Sparse lexical retrieval based on keyword overlap |
| Dense Retrieval | Embedding-based semantic retrieval using MiniLM sentence embeddings |
| Hybrid Retrieval | Candidate merging from BM25 and dense retrieval using reciprocal-rank style scoring |

The dense retriever uses normalized embeddings and FAISS inner-product search.

### Reranking

For reranked configurations, the system first retrieves a larger candidate pool and then applies a cross-encoder reranker.

The reranker scores each `(question, passage)` pair jointly, then reorders the candidate passages. The final top-5 passages are sent to the generator.

```text
Retriever candidate pool → Cross-encoder reranker → Final top-5 passages
```

### Prompting

We compare several prompt styles:

| Prompt Type | Description |
|---|---|
| Standard | Answer using the provided context |
| Grounded | Answer only if supported by context; otherwise say "Not enough information" |
| Evidence-based | Encourage evidence-like output |
| Short-grounded | Return the shortest supported answer span |

Short-grounded prompting is especially aligned with SQuAD because gold answers are usually short extractive spans.

### Generator

The generator is fixed across configurations:

```text
google/flan-t5-base
```

We use a fixed lightweight open-source model to keep the comparison reproducible and to isolate the effects of retrieval, reranking, and prompting.

## Evaluation

The project uses automatic metrics, oracle-context diagnostics, manual faithfulness review, and error analysis.

### Retrieval Metric

| Metric | Meaning |
|---|---|
| Recall@5 | Whether the gold answer string appears in one of the top five retrieved chunks |

Recall@5 measures evidence access. It does not guarantee that the generator will produce the correct answer.

### Answer Quality Metrics

| Metric | Meaning |
|---|---|
| Exact Match | Whether the generated answer exactly matches the gold answer after normalization |
| Token F1 | Token-level overlap between generated answer and gold answer |

These metrics are useful for SQuAD-style QA, but they do not fully measure faithfulness.

### Oracle-Context Check

The oracle-context experiment gives the generator the gold context directly instead of retrieved passages.

This diagnostic helps estimate how much performance is lost because of retrieval, context selection, truncation, prompt construction, or evidence use.

### Manual Faithfulness Review

A small manual review was conducted to check whether generated answers were supported by retrieved evidence.

| Score | Meaning |
|---:|---|
| 1.0 | Fully supported by retrieved context |
| 0.5 | Partially supported, incomplete, vague, or overly conservative |
| 0.0 | Unsupported, incorrect, hallucinated, or failed |

Manual error types include:

- `over_refusal`
- `wrong_answer`
- `correct_supported`
- `format_mismatch`
- `retrieval_failure`
- `partially_supported`

## Main Results

### Retrieval Results

Retrieval was evaluated over the 2,000-example SQuAD validation subset.

| Method | Recall@5 |
|---|---:|
| BM25 | 0.7900 |
| Dense Retrieval | 0.8685 |
| Hybrid Retrieval | 0.9125 |
| BM25 + Rerank | 0.8810 |
| Dense + Rerank | 0.9370 |
| Hybrid + Rerank | 0.9730 |

The strongest evidence access came from **Hybrid + Rerank**, with Recall@5 = **0.9730**.

### Answer Generation Results

Answer generation was evaluated on a smaller generation subset because running the generator across all configurations is computationally expensive.

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

The best automatic answer quality came from **Dense + Short Grounded**, with EM = **0.135** and F1 = **0.170**.

This shows that better retrieval does not automatically produce better generated answers. Retrieval provides access to evidence, but the generator still has to extract and format the answer correctly.

### Oracle-Context Results

| Configuration | EM | F1 |
|---|---:|---:|
| Oracle Context + Short Grounded | 0.695 | 0.831 |
| Oracle Context + Grounded | 0.660 | 0.786 |

Oracle-context performance is much higher than normal RAG performance. This suggests that FLAN-T5 can answer many questions when the correct evidence is clearly provided, but the full RAG pipeline is limited by context selection, prompt construction, prompt truncation, and evidence use.

### Manual Faithfulness Results

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

The strongest manual faithfulness score came from **Dense + Rerank + Short Grounded**.

The most common manual error type was **over-refusal**, where the model says "Not enough information" even though the retrieved evidence contains enough information to answer. This shows a tradeoff: grounded prompting can reduce unsupported generation, but it can also make the model too conservative.

## Key Findings

1. **Hybrid + Rerank maximized evidence access.**  
   It achieved the strongest Recall@5.

2. **Best retrieval did not produce the best EM/F1.**  
   Strong retrieval improves evidence access, but generation quality still depends on prompt construction and evidence use.

3. **Short-grounded prompting worked best for SQuAD-style answer quality.**  
   SQuAD answers are usually short extractive spans, so concise answer prompting aligns better with EM/F1.

4. **Manual faithfulness and EM/F1 diverged.**  
   EM/F1 measure string overlap, while faithfulness measures evidence support.

5. **Oracle-context results showed a large RAG pipeline gap.**  
   When the correct context was directly provided, generation quality was much higher.

6. **Over-refusal was the most common manual error type.**  
   Grounded prompting made the model more conservative, sometimes too conservative.

## Demo

The Gradio demo is designed as a **RAG faithfulness inspection tool**, not a production chatbot.

The intended user is a RAG developer or evaluator who wants to inspect whether generated answers are supported by retrieved evidence.

The demo shows:

- generated answer,
- gold answer when available,
- whether the gold answer appears in retrieved evidence,
- retrieval scores,
- automatic faithfulness diagnosis,
- retrieved evidence passages.

The prepared demo examples include:

1. an **over-refusal failure case**, where the retrieved evidence contains the gold answer but the model says "Not enough information";
2. a **successful grounded-answer case**, where the generated answer contains the gold answer and is supported by retrieved evidence.

The bottom of the demo interface also includes a short guide explaining how to adapt the demo to another QA corpus by replacing the dataset-loading block with a custom table containing `question`, `context`, and `answer` columns.

### Run the Demo

```bash
python demo_app.py
```

The first run may take several minutes because pretrained models are downloaded and the retrieval index is built.

## Repository Structure

```text
STAT5293-Project/
├── README.md
├── requirements.txt
├── demo_app.py
├── 5293_Final_Project.ipynb
├── test_basic_pipeline.py
│
├── figures/
│   ├── retrieval_performance_comparison.png
│   ├── answer_quality_comparison.png
│   └── manual_faithfulness_comparison.png
│
└── results/
    ├── answer_quality_summary.csv
    ├── manual_faithfulness_review_labeled.csv
    ├── oracle_context_results.csv
    ├── oracle_context_summary.csv
    ├── rag_generation_results.csv
    ├── representative_error_cases.csv
    └── selected_error_cases.csv
```

If a recorded demo video is included, it can be placed under:

```text
demo/
└── rag_faithfulness_demo.mp4
```

## Setup

Install dependencies:

```bash
pip install -r requirements.txt
```

Main dependencies include:

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

For faster execution, GPU runtime is recommended.

## Running the Experiments

Open the notebook:

```text
5293_Final_Project.ipynb
```

The notebook includes:

1. dataset loading,
2. document chunking,
3. retrieval index construction,
4. BM25 retrieval,
5. dense retrieval,
6. hybrid retrieval,
7. cross-encoder reranking,
8. prompt construction,
9. FLAN-T5 answer generation,
10. automatic evaluation,
11. oracle-context evaluation,
12. visualization,
13. manual faithfulness review,
14. error analysis,
15. conclusion and future work.

The notebook was designed to run in Google Colab.

## Basic Tests

This repository includes a basic validation script:

```text
test_basic_pipeline.py
```

It checks that key result files exist and contain expected columns, including:

- `config_name`
- `em`
- `f1`
- `faithfulness_label`
- `error_type`

Run the tests with:

```bash
pip install pytest
pytest
```

These tests are not a full production test suite, but they help verify that the saved experiment outputs are present and readable.

## Limitations

This project has several limitations:

- The dataset is based on SQuAD, where answers are usually short extractive spans.
- EM and F1 may penalize faithful answers that do not exactly match the gold string.
- The generator model, `google/flan-t5-base`, is lightweight and less capable than larger instruction-tuned models.
- Manual faithfulness review is based on a small sample and should be interpreted as qualitative diagnostic evidence.
- The hybrid retrieval method uses simple rank-based merging rather than learned fusion.
- The demo builds the retrieval index at startup, so the first run may be slow.
- The evaluation is not a production benchmark and does not claim state-of-the-art RAG performance.

## Future Work

Future improvements could include:

- tuning refusal behavior to reduce over-refusal,
- improving context construction and passage ordering,
- adding citation or span extraction to force more direct grounding,
- expanding manual faithfulness review,
- using multiple annotators and reporting inter-annotator agreement,
- evaluating on more realistic multi-document or open-domain QA datasets,
- improving hybrid retrieval with learned fusion or score calibration,
- testing stronger generator models.

## Troubleshooting

### The demo takes a long time to start

The first run downloads pretrained models from Hugging Face and builds the FAISS retrieval index. This is expected.

### CUDA is not available

The code falls back to CPU automatically, but generation and reranking will be slower.

### FAISS installation error

Try upgrading pip first:

```bash
pip install --upgrade pip
pip install faiss-cpu
```

### Hugging Face model download issues

Restart the runtime and rerun the app or notebook. The models used are:

```text
sentence-transformers/all-MiniLM-L6-v2
cross-encoder/ms-marco-MiniLM-L-6-v2
google/flan-t5-base
```

### Notebook rendering error on GitHub

If GitHub reports a notebook widget metadata error, remove the `metadata.widgets` field from the notebook JSON and re-upload the cleaned notebook.

### Results differ slightly

Some results may differ slightly depending on hardware, package versions, and runtime settings. The dataset shuffle uses `seed=42`, and saved result CSV files are included in the `results/` folder for reference.

### Out-of-memory error

Reduce generation evaluation size or run the notebook in Google Colab with GPU enabled.

## Authors

- Kevin Ma
- Joseph Yeung
- Zenan Luo


