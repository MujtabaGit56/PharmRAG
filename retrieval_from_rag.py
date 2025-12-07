import os
import re
import json
from typing import List, Dict, Tuple, Any
import numpy as np
import faiss
import torch
from transformers import AutoTokenizer, AutoModel, AutoModelForSequenceClassification
from drug_detection import build_comprehensive_drug_index, detect_drug_from_query_robust

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ============================================================================
# BGE-MEDICAL-WITH-MEDIMAVEN-RERANKER
# ============================================================================

MODEL_NAME = "aleynahukmet/bge-medical-small-en-v1.5"
RERANKER_NAME = "dranreb1660/medimaven-reranker-bge-cross-encoder"

EMB_DIR = r"C:\Users\mujta\Desktop\Project Machine Learning\PharmRAG\Project-Files\Attempt 6\Misc\embeddings\bge-medical"
EMB_FILE = os.path.join(EMB_DIR, "embeddings-bge-medical-final-800.npy")
META_FILE = os.path.join(EMB_DIR, "metadata-bge-medical-final-800.jsonl")
INDEX_FILE = os.path.join(EMB_DIR, "index-faiss-ip.bin")

# ============================================================================
# BGE-GENERAL-WITH-MEDIMAVEN-RERANKER
# ============================================================================

# MODEL_NAME = "BAAI/bge-small-en-v1.5"
# RERANKER_NAME = "dranreb1660/medimaven-reranker-bge-cross-encoder"
#
# EMB_DIR = r"C:\Users\mujta\Desktop\Project Machine Learning\PharmRAG\Project-Files\Attempt 6\Misc\embeddings\bge-general"
# EMB_FILE = os.path.join(EMB_DIR, "embeddings-bge-general-final-800.npy")
# META_FILE = os.path.join(EMB_DIR, "metadata-bge-general-final-800.jsonl")
# INDEX_FILE = os.path.join(EMB_DIR, "index-faiss-ip.bin")

# ============================================================================

TOP_K = 8
FAISS_K = 250  # Increased to get more candidates
RERANK_BATCH_SIZE = 32

SECTION_PRIORITIES = {
    "usage": ["indications_and_usage", "description", "patient_medication_information"],
    "dosage": ["dosage_and_administration", "patient_medication_information", "warnings"],
    "side_effects": ["adverse_reactions", "warnings", "boxed_warning", "stop_use"],
    "interactions": ["drug_interactions", "warnings", "precautions"],
    "pregnancy": ["pregnancy_or_breast_feeding", "contraindications", "warnings"],
    "warnings": ["boxed_warning", "warnings", "contraindications", "precautions"],
    "boxed_warning": ["boxed_warning", "warnings", "contraindications"],
}


def classify_query_intent(query: str) -> str:
    q = query.lower()

    if "boxed warning" in q or "black box" in q:
        return "boxed_warning"

    if any(kw in q for kw in ["what is", "used for", "treat", "for", "help with"]):
        return "usage"
    if any(kw in q for kw in ["dose", "how much", "how many", "twice", "missed"]):
        return "dosage"
    if any(kw in q for kw in ["side effect", "reaction", "dizzy", "nausea", "rash"]):
        return "side_effects"
    if any(kw in q for kw in ["interact", "with", "take with", "alcohol"]):
        return "interactions"
    if any(kw in q for kw in ["pregnant", "pregnancy", "breastfeed", "nursing"]):
        return "pregnancy"
    if any(kw in q for kw in ["warning", "danger", "avoid", "safe"]):
        return "warnings"

    return "general"

def load_metadata(path: str) -> List[Dict]:
    metas = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            metas.append(json.loads(line))
    return metas


def load_index(path: str) -> faiss.Index:
    if os.path.exists(path):
        index = faiss.read_index(path)
        return index
    raise FileNotFoundError(f"Index not found: {path}")


def load_model(name: str):
    tok = AutoTokenizer.from_pretrained(name)
    model = AutoModel.from_pretrained(name)
    model.to(DEVICE)
    model.eval()
    return tok, model


def load_reranker(name: str):
    tok = AutoTokenizer.from_pretrained(name)
    model = AutoModelForSequenceClassification.from_pretrained(name)
    model.to(DEVICE)
    model.eval()
    return tok, model

def mean_pool(last_hidden_state, attention_mask):
    mask = attention_mask.unsqueeze(-1)
    masked = last_hidden_state * mask
    summed = masked.sum(dim=1)
    counts = mask.sum(dim=1).clamp(min=1e-9)
    return summed / counts


def embed_query(texts: List[str], tokenizer, model, max_length=512) -> np.ndarray:
    with torch.no_grad():
        enc = tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        ).to(DEVICE)
        out = model(**enc)
        pooled = mean_pool(out.last_hidden_state, enc["attention_mask"])
        pooled = torch.nn.functional.normalize(pooled, p=2, dim=1)
    return pooled.cpu().numpy().astype("float32")

def build_drug_index(metas: List[Dict[str, Any]]) -> Dict[str, set]:
    return build_comprehensive_drug_index(metas, verbose=False)


def detect_drug_from_query(query: str, drug_index: Dict[str, set], metas: List[Dict]) -> str | None:
    return detect_drug_from_query_robust(query, drug_index, verbose=False)

def cross_encoder_rerank(
        query: str,
        candidates: List[Dict[str, Any]],
        reranker,
        top_k: int,
) -> List[Tuple[float, Dict[str, Any]]]:
    tok, model = reranker
    texts = [c["text"] for c in candidates]
    metas = [c["meta"] for c in candidates]
    scores = []

    with torch.no_grad():
        for i in range(0, len(texts), RERANK_BATCH_SIZE):
            batch = texts[i:i + RERANK_BATCH_SIZE]
            enc = tok(
                [query] * len(batch),
                batch,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt",
            ).to(DEVICE)
            logits = model(**enc).logits.squeeze(-1)
            scores.extend(logits.cpu().numpy().tolist())

    out = list(zip(scores, metas))
    out.sort(key=lambda x: x[0], reverse=True)
    return out[:top_k]

def search(
        query: str,
        index: faiss.Index,
        metas: List[Dict[str, Any]],
        tokenizer,
        model,
        reranker=None,
        top_k: int = TOP_K,
        faiss_k: int = FAISS_K,
        drug_index: Dict[str, set] = None,
) -> List[Tuple[float, Dict[str, Any]]]:
    query_intent = classify_query_intent(query)

    detected_drug = None
    if drug_index:
        detected_drug = detect_drug_from_query(query, drug_index, metas)

    q_emb = embed_query([query], tokenizer, model)
    scores, idxs = index.search(q_emb, faiss_k)

    if scores.ndim == 2:
        scores, idxs = scores[0], idxs[0]

    candidates = []
    for score, i in zip(scores, idxs):
        if i == -1:
            continue

        meta = metas[i]
        text = meta.get("text", "").strip()
        if not text:
            continue
        final_score = float(score)

        if detected_drug:
            drug_name = (meta.get("drug_name") or "").lower()
            generic_name = (meta.get("generic_name") or "").lower()
            brand_name = (meta.get("brand_name") or "").lower()

            if (detected_drug in drug_name or
                    detected_drug in generic_name or
                    detected_drug in brand_name):
                final_score += 10.0
            else:
                final_score -= 5.0

        section = meta.get("section", "")
        if query_intent in SECTION_PRIORITIES:
            priority_sections = SECTION_PRIORITIES[query_intent]
            if section in priority_sections:
                rank = priority_sections.index(section)
                boost = (len(priority_sections) - rank) * 3.0
                final_score += boost

        if meta.get("audience") == "consumer":
            final_score += 1.0

        candidates.append({
            "score": final_score,
            "meta": meta,
            "text": text,
        })

    if not candidates:
        return []

    candidates.sort(key=lambda c: c["score"], reverse=True)
    top_candidates = candidates[:min(30, len(candidates))]

    if reranker:
        return cross_encoder_rerank(query, top_candidates, reranker, top_k)

    return [(c["score"], c["meta"]) for c in top_candidates[:top_k]]

def strip_header(text: str) -> str:
    if text.startswith("[") and "]" in text[:300]:
        return text.split("]", 1)[1].strip()
    return text


def format_result(rank: int, score: float, meta: Dict[str, Any]) -> str:
    text = strip_header(meta.get("text", ""))
    section = meta.get("section", "?")
    drug = meta.get("drug_name", "?")
    audience = meta.get("audience", "?")

    if len(text) > 800:
        text = text[:800] + "..."

    output = f"\n{rank}. Score: {score:.3f}\n"
    output += f"   Drug: {drug}\n"
    output += f"   Section: {section} ({audience})\n"
    output += f"   {text}\n"

    return output

def demo():
    metas = load_metadata(META_FILE)
    drug_index = build_drug_index(metas)
    index = load_index(INDEX_FILE)
    tokenizer, model = load_model(MODEL_NAME)
    reranker = load_reranker(RERANKER_NAME)

    print("\n" + "=" * 80)
    print("Local Pharmacy RAG System")
    print("=" * 80)

    while True:
        q = input("\nEnter your question (or 'quit'): ").strip()
        if q.lower() in ["quit", "exit", "q"]:
            break

        if not q:
            continue

        hits = search(
            q,
            index=index,
            metas=metas,
            tokenizer=tokenizer,
            model=model,
            reranker=reranker,
            top_k=TOP_K,
            faiss_k=FAISS_K,
            drug_index=drug_index,
        )

        if not hits:
            print("\nNo results found.")
            continue

        print("\n" + "-" * 80)
        print("Top Results:")
        print("-" * 80)

        for rank, (score, meta) in enumerate(hits, start=1):
            print(format_result(rank, score, meta))


if __name__ == "__main__":
    demo()