import os
import json
from typing import List, Dict, Tuple, Any
import time

import retrieval_from_rag as retrieval_mod
import llm_inference as llm_mod
from drug_detection import detect_drug_from_query_robust
from difflib import SequenceMatcher

QUESTIONS_FILE = "best-150-questions.txt"
OUTPUT_FILE = "150-rag-answers.json"

TOP_K_REFERENCE = 2  # Number of top retrieved chunks to store


def strip_header(text: str) -> str:
    if text.startswith("[") and "]" in text[:300]:
        return text.split("]", 1)[1].strip()
    return text


def extract_drug_id(question: str, drug_index):
    detected = detect_drug_from_query_robust(question, drug_index, verbose=False)
    return detected if detected else "unknown"


def are_similar(a: str, b: str, threshold: float = 0.995) -> bool:
    if not a or not b:
        return False
    return SequenceMatcher(None, a, b).ratio() >= threshold


def format_retrieved_passages(
        hits: List[Tuple[float, Dict[str, Any]]],
        top_k: int = 3
) -> List[Dict[str, str]]:
    unique_texts: List[str] = []

    for score, meta in hits:
        text = meta.get("text", "").strip()
        text = strip_header(text)
        if not text:
            continue

        if any(are_similar(text, existing) for existing in unique_texts):
            continue

        unique_texts.append(text)

        if len(unique_texts) >= top_k:
            break
    return [{"id": idx, "text": t} for idx, t in enumerate(unique_texts)]


def process_single_question(
        question: str,
        question_id: int,
        rag_components: Dict[str, Any]
) -> Dict[str, Any]:
    """Process a single question and return results."""
    print(f"  Processing Q{question_id}: {question[:80]}...")

    # Get retrieved passages
    hits = retrieval_mod.search(
        query=question,
        index=rag_components['index'],
        metas=rag_components['metas'],
        tokenizer=rag_components['tokenizer'],
        model=rag_components['model'],
        reranker=rag_components['reranker'],
        top_k=10,
        faiss_k=250,
        drug_index=rag_components['drug_index']
    )

    retrieved_passages = format_retrieved_passages(hits, top_k=TOP_K_REFERENCE)

    # Get LLM answer using the simplified interface
    # llm_mod.rag_answer() will use its own global components
    predicted_answer = llm_mod.rag_answer(user_query=question)

    # Extract drug identifier
    drug_id = extract_drug_id(question, rag_components['drug_index'])

    return {
        "id": question_id,
        "drug_id": drug_id,
        "question": question,
        "predicted_answer": predicted_answer,
        "retrieved_rag_passages": retrieved_passages
    }


def load_questions(filepath: str) -> List[str]:
    with open(filepath, 'r', encoding='utf-8') as f:
        questions = [line.strip() for line in f if line.strip()]
    return questions


def save_results(results: List[Dict[str, Any]], filepath: str):
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"✓ Saved {len(results)} results to: {filepath}")


def main():
    print("=" * 80)
    print("RAG ANSWER GENERATION")
    print("=" * 80)

    # Initialize LLM system first (this loads all components internally)
    print("\n1. Initializing LLM inference system...")
    init_status = llm_mod.initialize_rag_system()
    if "failed" in init_status.lower():
        print(f"ERROR: {init_status}")
        return

    # Load RAG components for retrieval analysis
    print("\n2. Loading RAG components for passage retrieval...")
    metas = retrieval_mod.load_metadata(retrieval_mod.META_FILE)
    drug_index = retrieval_mod.build_drug_index(metas)
    index = retrieval_mod.load_index(retrieval_mod.INDEX_FILE)
    tokenizer, model = retrieval_mod.load_model(retrieval_mod.MODEL_NAME)
    reranker = retrieval_mod.load_reranker(retrieval_mod.RERANKER_NAME)

    rag_components = {
        'index': index,
        'metas': metas,
        'tokenizer': tokenizer,
        'model': model,
        'reranker': reranker,
        'drug_index': drug_index
    }

    # Load questions
    print(f"\n3. Loading questions from {QUESTIONS_FILE}...")
    questions = load_questions(QUESTIONS_FILE)
    print(f"   Loaded {len(questions)} questions")

    print(f"\n4. Processing questions (saving top-{TOP_K_REFERENCE} retrieved chunks)...")
    results = []

    for i, question in enumerate(questions, 1):
        try:
            result = process_single_question(
                question=question,
                question_id=i,
                rag_components=rag_components
            )
            results.append(result)

            # Save after each question (in case of crash)
            if i % 5 == 0:  # Save every 5 questions
                save_results(results, OUTPUT_FILE)
                print(f"   Progress: {i}/{len(questions)} completed")

        except Exception as e:
            print(f"\n   ERROR on Q{i}: {str(e)}")
            results.append({
                "id": i,
                "drug_id": "error",
                "question": question,
                "predicted_answer": f"ERROR: {str(e)}",
                "retrieved_rag_passages": [],
                "error": str(e)
            })

    # Final save
    print(f"\n5. Saving final results...")
    save_results(results, OUTPUT_FILE)

    print("\n" + "=" * 80)
    print("GENERATION COMPLETE")
    print("=" * 80)
    print(f"Total questions: {len(questions)}")
    print(f"Successful: {len([r for r in results if 'error' not in r])}")
    print(f"Failed: {len([r for r in results if 'error' in r])}")
    print(f"\nOutput saved to: {OUTPUT_FILE}")
    print("\nNext step: Run metric calculation script on this file")


if __name__ == "__main__":
    main()