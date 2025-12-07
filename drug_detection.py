import re
from typing import List, Dict, Set, Tuple, Any

def build_comprehensive_drug_index(metas: List[Dict[str, Any]], verbose: bool = False) -> Dict[str, Any]:
    full_names: Dict[str, Set[int]] = {}
    token_index: Dict[str, Set[int]] = {}

    all_drug_names: Set[str] = set()

    for i, m in enumerate(metas):
        names = []
        for key in ["drug_name", "brand_name", "generic_name", "substance_name"]:
            val = (m.get(key) or "").strip()
            if val and len(val) >= 3:
                names.append(val)

        if not names:
            continue

        for name in names:
            normalized = " ".join(name.lower().split())

            full_names.setdefault(normalized, set()).add(i)
            all_drug_names.add(normalized)

            tokens = re.findall(r"[a-z][a-z0-9_-]*", normalized)
            for token in tokens:
                if len(token) >= 4:
                    token_index.setdefault(token, set()).add(i)

    if verbose:
        print(f"Indexed {len(all_drug_names)} unique drug names")
        print(f"Created token index with {len(token_index)} tokens")

    return {
        "full_names": full_names,
        "token_index": token_index,
        "all_drug_names": all_drug_names,
    }

STOPWORDS = {
    # Question words
    "what", "when", "where", "which", "who", "whom", "whose", "why", "how",

    # Common verbs
    "can", "could", "would", "should", "will", "shall", "may", "might", "must",
    "do", "does", "did", "have", "has", "had", "is", "are", "was", "were",
    "take", "taking", "took", "taken", "use", "using", "used", "get", "getting",

    # Prepositions
    "for", "with", "about", "against", "between", "into", "through", "during",
    "before", "after", "above", "below", "from", "up", "down", "in", "out",
    "on", "off", "over", "under",

    # Pronouns and articles
    "i", "you", "he", "she", "it", "we", "they", "me", "him", "her", "us", "them",
    "my", "your", "his", "her", "its", "our", "their",
    "the", "a", "an", "this", "that", "these", "those",

    # Common medical query words (not drug names)
    "medication", "medicine", "drug", "pill", "tablet", "capsule", "dose",
    "side", "effect", "effects", "reaction", "symptom", "symptoms",
    "doctor", "prescribed", "prescription",

    # Other common words
    "there", "here", "some", "any", "many", "much", "more", "most",
    "something", "anything", "everything", "nothing", "label", "labels",
}

# Common non-drug terms that might appear in drug names but aren't drugs themselves
NOISE_WORDS = {
    "extra", "strength", "maximum", "regular", "plus", "advanced", "complete",
    "relief", "oral", "topical", "injection", "cream", "ointment", "liquid",
    "tablet", "capsule", "extended", "release", "control", "label", "labels",
}

# ============================================================================
# DRUG DETECTION LOGIC
# ============================================================================

def extract_ngrams(text: str, max_n: int = 4) -> List[str]:
    words = text.lower().split()
    ngrams = []

    for n in range(1, min(max_n + 1, len(words) + 1)):
        for i in range(len(words) - n + 1):
            ngram = " ".join(words[i:i + n])
            ngrams.append(ngram)

    return ngrams


def detect_drugs_from_query(
        query: str,
        drug_index: Dict[str, Any],
        min_token_length: int = 4,
        max_results: int = 3
) -> List[Tuple[str, float]]:

    full_names = drug_index["full_names"]
    token_index = drug_index["token_index"]
    all_drug_names = drug_index["all_drug_names"]

    query_lower = query.lower()

    ngrams = extract_ngrams(query_lower, max_n=4)

    candidates: Dict[str, float] = {}

    for ngram in ngrams:
        if ngram in full_names:
            confidence = len(ngram.split()) * 10.0
            candidates[ngram] = max(candidates.get(ngram, 0), confidence)

    if not candidates:
        for drug_name in all_drug_names:
            if drug_name in query_lower or query_lower in drug_name:
                overlap = len(set(drug_name.split()) & set(query_lower.split()))
                confidence = overlap * 3.0
                candidates[drug_name] = max(candidates.get(drug_name, 0), confidence)

    if not candidates:
        tokens = re.findall(r"[a-z][a-z0-9_-]+", query_lower)

        for token in tokens:
            if len(token) < min_token_length or token in STOPWORDS:
                continue

            if token in NOISE_WORDS:
                continue

            if token in token_index:
                indices = token_index[token]

                if len(indices) < 100:
                    confidence = 2.0
                elif len(indices) < 500:
                    confidence = 1.0
                else:
                    continue
                candidates[token] = max(candidates.get(token, 0), confidence)

    sorted_candidates = sorted(
        candidates.items(),
        key=lambda x: (-x[1], -len(x[0]))
    )

    return sorted_candidates[:max_results]

def filter_drug_candidates(
        candidates: List[Tuple[str, float]],
        query: str,
        min_confidence: float = 1.5
) -> List[str]:

    validated = []
    query_lower = query.lower()

    for drug_name, confidence in candidates:
        if confidence < min_confidence:
            continue

        if drug_name in STOPWORDS:
            continue

        common_query_words = {
            "pain", "cold", "flu", "fever", "allergy", "cough", "sleep",
            "relief", "control", "treatment", "therapy"
        }
        if drug_name in common_query_words:
            continue

        validated.append(drug_name)

    return validated

def detect_drug_from_query_robust(
        query: str,
        drug_index: Dict[str, Any],
        verbose: bool = False
) -> str | None:

    candidates = detect_drugs_from_query(query, drug_index, max_results=5)

    if verbose:
        print(f"[Drug Detection] Raw candidates: {candidates}")

    validated = filter_drug_candidates(candidates, query, min_confidence=2.0)

    if verbose:
        print(f"[Drug Detection] Validated: {validated}")

    if not validated:
        return None

    return validated[0]

def test_drug_detection(metas: List[Dict[str, Any]]):

    print("Building drug index...")
    drug_index = build_comprehensive_drug_index(metas, verbose=True)

    test_queries = [
        ("What is spironolactone used for?", "spironolactone"),
        ("Side effects of ibuprofen", "ibuprofen"),
        ("Can I take Tylenol with aspirin?", "tylenol or aspirin"),
        ("Lipitor 20mg dosage", "lipitor"),
        ("metformin side effects", "metformin"),
        ("Is Advil safe during pregnancy?", "advil"),

        ("What are common side effects of blood pressure medication?", None),
        ("How do I take my medicine?", None),
        ("My doctor prescribed something for pain", None),
        ("I have a headache", None),

        ("Extra Strength Tylenol dosage", "tylenol"),
        ("acetaminophen 500mg", "acetaminophen"),
        ("Can I drink alcohol with my blood thinner?", None or "blood thinner"),
    ]

    print("\n" + "=" * 80)
    print("DRUG DETECTION TEST RESULTS")
    print("=" * 80)

    for query, expected in test_queries:
        detected = detect_drug_from_query_robust(query, drug_index, verbose=False)
        status = "✓" if detected else "✗"

        print(f"\n{status} Query: {query}")
        print(f"  Expected: {expected}")
        print(f"  Detected: {detected}")