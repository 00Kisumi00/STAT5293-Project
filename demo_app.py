
import numpy as np
import pandas as pd
import gradio as gr
import faiss
import torch

from datasets import load_dataset
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer, CrossEncoder
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


# =========================
# 1. Load dataset
# =========================

raw_dataset = load_dataset("squad", split="validation")
dataset = raw_dataset.shuffle(seed=42).select(range(2000))

df = pd.DataFrame({
    "question": dataset["question"],
    "context": dataset["context"],
    "answer": [a["text"][0] if len(a["text"]) > 0 else "" for a in dataset["answers"]]
})

documents = list(df["context"].drop_duplicates())


# =========================
# 2. Chunk documents
# =========================

def chunk_text(text, chunk_size=120, overlap=30):
    words = text.split()
    chunks = []
    step = chunk_size - overlap

    for i in range(0, len(words), step):
        chunk = " ".join(words[i:i + chunk_size])
        if len(chunk.strip()) > 0:
            chunks.append(chunk)

    return chunks


chunks = []

for doc_id, doc in enumerate(documents):
    doc_chunks = chunk_text(doc)

    for chunk_id, chunk in enumerate(doc_chunks):
        chunks.append({
            "doc_id": doc_id,
            "chunk_id": chunk_id,
            "text": chunk
        })

chunks_df = pd.DataFrame(chunks)


# =========================
# 3. Build BM25 retriever
# =========================

tokenized_corpus = [text.lower().split() for text in chunks_df["text"]]
bm25 = BM25Okapi(tokenized_corpus)


def retrieve_bm25(query, top_k=10):
    tokenized_query = query.lower().split()
    scores = bm25.get_scores(tokenized_query)

    top_indices = np.argsort(scores)[::-1][:top_k]

    results = chunks_df.iloc[top_indices].copy()
    results["score"] = scores[top_indices]
    results["source"] = "BM25"

    return results


# =========================
# 4. Build dense retriever
# =========================

embed_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

chunk_embeddings = embed_model.encode(
    chunks_df["text"].tolist(),
    show_progress_bar=True,
    convert_to_numpy=True,
    normalize_embeddings=True
)

dimension = chunk_embeddings.shape[1]
dense_index = faiss.IndexFlatIP(dimension)
dense_index.add(chunk_embeddings)


def retrieve_dense(query, top_k=10):
    query_embedding = embed_model.encode(
        [query],
        convert_to_numpy=True,
        normalize_embeddings=True
    )

    scores, indices = dense_index.search(query_embedding, top_k)

    results = chunks_df.iloc[indices[0]].copy()
    results["score"] = scores[0]
    results["source"] = "Dense"

    return results


# =========================
# 5. Hybrid retrieval + reranking
# =========================

reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")


def retrieve_hybrid_reranked(query, top_k=5, pool_k=20):
    bm25_results = retrieve_bm25(query, top_k=pool_k).copy()
    dense_results = retrieve_dense(query, top_k=pool_k).copy()

    bm25_results["source"] = "BM25"
    dense_results["source"] = "Dense"

    bm25_results["rank"] = np.arange(1, len(bm25_results) + 1)
    dense_results["rank"] = np.arange(1, len(dense_results) + 1)

    bm25_results["hybrid_score"] = 1 / bm25_results["rank"]
    dense_results["hybrid_score"] = 1 / dense_results["rank"]

    candidates = pd.concat([bm25_results, dense_results], axis=0)

    candidates = (
        candidates
        .groupby(["doc_id", "chunk_id"], as_index=False)
        .agg({
            "text": "first",
            "score": "max",
            "source": lambda x: "+".join(sorted(set(x))),
            "rank": "min",
            "hybrid_score": "sum"
        })
    )

    candidates = candidates.sort_values("hybrid_score", ascending=False)

    pairs = [(query, text) for text in candidates["text"].tolist()]
    rerank_scores = reranker.predict(pairs)

    candidates = candidates.copy()
    candidates["rerank_score"] = rerank_scores
    candidates = candidates.sort_values("rerank_score", ascending=False)

    return candidates.head(top_k)


# =========================
# 6. Generator
# =========================

generator_model_name = "google/flan-t5-base"

tokenizer = AutoTokenizer.from_pretrained(generator_model_name)
generation_model = AutoModelForSeq2SeqLM.from_pretrained(generator_model_name)

device = "cuda" if torch.cuda.is_available() else "cpu"
generation_model = generation_model.to(device)


def build_grounded_prompt(question, contexts):
    context_text = "\n\n".join(contexts)

    return f"""
You must answer the question only using the provided context.
If the answer is not supported by the context, say "Not enough information."

Context:
{context_text}

Question:
{question}

Grounded answer:
"""


def generate_answer(prompt, max_new_tokens=128):
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=512
    ).to(device)

    with torch.no_grad():
        outputs = generation_model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False
        )

    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return answer


# =========================
# 7. Demo function
# =========================

def find_gold_answer(question):
    matched = df[df["question"].str.strip().str.lower() == question.strip().lower()]

    if len(matched) > 0:
        return matched.iloc[0]["answer"]

    return "No gold answer available for this custom question."


def rag_demo(question):
    try:
        if question is None or question.strip() == "":
            return "Please enter a question.", "", ""

        retrieved = retrieve_hybrid_reranked(
            query=question,
            top_k=5,
            pool_k=20
        )

        contexts = retrieved["text"].tolist()
        prompt = build_grounded_prompt(question, contexts)
        answer = generate_answer(prompt)
        gold_answer = find_gold_answer(question)

        evidence_blocks = []

        for rank, (_, row) in enumerate(retrieved.iterrows(), start=1):
            evidence_blocks.append(
                f"""### Passage {rank}
**Source:** {row["source"]}  
**Retrieval score:** {row["score"]:.4f}  
**Hybrid score:** {row.get("hybrid_score", 0):.4f}  
**Rerank score:** {row["rerank_score"]:.4f}

{row["text"]}
"""
            )

        evidence_text = "\n\n---\n\n".join(evidence_blocks)

        return answer, gold_answer, evidence_text

    except Exception as e:
        return (
            "An error occurred while running the RAG pipeline.",
            "No gold answer available.",
            f"Error details: {str(e)}"
        )


# =========================
# 8. Gradio app
# =========================

example_questions = df["question"].sample(5, random_state=7).tolist()

demo = gr.Interface(
    fn=rag_demo,
    inputs=gr.Textbox(
        label="Question",
        placeholder="Enter a question here..."
    ),
    outputs=[
        gr.Textbox(label="Generated Answer"),
        gr.Textbox(label="Gold Answer"),
        gr.Markdown(label="Retrieved Evidence")
    ],
    title="RAG Faithfulness Demo",
    description=(
        "This demo uses Hybrid Retrieval + Reciprocal-Rank Candidate Merging "
        "+ Cross-Encoder Reranking + Grounded Prompting. "
        "It retrieves relevant passages from a shuffled SQuAD-based document collection, "
        "generates an answer using the retrieved evidence, and displays the gold answer "
        "when the input question comes from the SQuAD subset."
    ),
    examples=example_questions
)


if __name__ == "__main__":
    demo.launch(share=True)
