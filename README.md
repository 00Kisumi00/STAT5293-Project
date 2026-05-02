# Improving Faithfulness in RAG

This project studies how retrieval, reranking, and prompting strategies affect answer quality and faithfulness in a simple Retrieval-Augmented Generation (RAG) question answering system.

The system uses a shuffled subset of 2,000 SQuAD validation examples as the document collection. It compares BM25 sparse retrieval, dense embedding retrieval, hybrid retrieval, cross-encoder reranking, and multiple prompting strategies.

## Project Motivation

Large language models can generate fluent answers, but they may also produce information that is not supported by evidence. RAG is a common approach for reducing this problem by retrieving relevant passages and using them as context for answer generation.

However, the final answer quality depends on several design choices:

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
