import os
from typing import List, Tuple, Dict, Any
import google.generativeai as genai
import retrieval_from_rag as rag

TOP_K = 10
FAISS_K = 250

EMB_DIR = rag.EMB_DIR
META_FILE = rag.META_FILE
INDEX_FILE = rag.INDEX_FILE
MODEL_NAME = rag.MODEL_NAME
RERANKER_NAME = rag.RERANKER_NAME

_metas = None
_drug_index = None
_index = None
_tokenizer = None
_model = None
_reranker = None
_gemini_model = None
_initialized = False

def init_gemini():
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "ERROR: GEMINI_API_KEY environment variable not set.\n"
        )

    # api_key = "---" # comment out the above code if you want to use your own api-key

    genai.configure(api_key=api_key)

    generation_config = genai.GenerationConfig(
        temperature=0.3,
        top_p=0.9,
        max_output_tokens=2048,
    )

    model_name = "gemini-2.5-flash"
    model = genai.GenerativeModel(model_name, generation_config=generation_config)
    return model

SYSTEM_PROMPT = """
You are a pharmacy assistant that answers medication questions using ONLY the FDA label snippets provided.
Your responses must be short, clear, and safety-focused.

OUTPUT FORMAT RULES (MANDATORY):
• Final answer should usually be about 6-8 lines total (NOT huge paragraphs).
• When the question is very narrow (e.g., a single instruction), still restate the key instruction AND add 2–4 additional lines of relevant context from the snippets (such as related directions, warnings, or safety advice).
• Use bullet points automatically whenever listing uses, warnings, side effects, or steps.
• Use a single concise paragraph only when the answer is explanatory (e.g., "why" questions).
• No headings like '###'; no long sections; no emojis.
• Every line must contain meaningful information—no filler.

SAFETY RULES:
• Always check ALL snippets for warnings, contraindications, boxed warnings, and serious risks.
• If any critical safety issue appears, mention it clearly in the answer.
• Never downplay or ignore FDA warnings.
• Do NOT guess or invent information not supported by the snippets.
• If the label does not answer the question, say so directly.

STYLE:
• Plain English, pharmacy-assistant tone.
• Direct, simple, and easy to read.
• If the question is about "what it is used for," prefer bullet points.
• If the question is about "why," give a compact paragraph (still ≤ 6 lines).

ALWAYS read all snippets before answering.
"""

def format_snippet(i: int, score: float, meta: Dict[str, Any]) -> str:
    raw_text = (meta.get("text") or "").strip()
    clean_text = rag.strip_header(raw_text)

    return (
        f"SNIPPET {i}:\n"
        f"Drug: {meta.get('drug_name', 'UNKNOWN')}\n"
        f"Section: {meta.get('section', 'UNKNOWN')}\n"
        f"Audience: {meta.get('audience', 'UNKNOWN')}\n"
        f"Score: {score:.3f}\n"
        f"{clean_text}"
    )


def build_context_block(hits: List[Tuple[float, Dict[str, Any]]], max_snippets: int = 10) -> str:
    lines = []

    by_section = {}
    for score, meta in hits[:max_snippets]:
        section = meta.get("section", "unknown")
        if section not in by_section:
            by_section[section] = []
        by_section[section].append((score, meta))

    critical_sections = [
        "boxed_warning",
        "contraindications",
        "warnings",
        "warnings_and_cautions",
        "indications_and_usage",
    ]

    i = 1
    for section in critical_sections:
        if section in by_section:
            for score, meta in by_section[section]:
                lines.append(format_snippet(i, score, meta))
                i += 1
            del by_section[section]

    for section, chunks in by_section.items():
        for score, meta in chunks:
            lines.append(format_snippet(i, score, meta))
            i += 1

    return "\n---\n".join(lines)

def extract_primary_drug_from_hits(hits: List[Tuple[float, Dict[str, Any]]]) -> str:
    if not hits:
        return "this medication"

    first_drug = (hits[0][1].get("drug_name") or "").strip()

    if first_drug:
        return first_drug

    counts = {}
    for _, meta in hits:
        name = (meta.get("drug_name") or "").strip()
        if name:
            counts[name] = counts.get(name, 0) + 1

    if counts:
        return max(counts.items(), key=lambda x: x[1])[0]

    return "this medication"

def build_adaptive_prompt(user_query: str, hits: List[Tuple[float, Dict[str, Any]]],
                          few_shot_examples: str = "") -> str:
    query_type = rag.classify_query_intent(user_query)
    context = build_context_block(hits, max_snippets=10)
    drug_name = extract_primary_drug_from_hits(hits)

    # Add few-shot examples if provided
    few_shot_section = ""
    if few_shot_examples and few_shot_examples.strip():
        few_shot_section = f"\nFEW-SHOT EXAMPLES (Follow this style):\n{few_shot_examples}\n"

    base = f"""{SYSTEM_PROMPT}
{few_shot_section}
USER QUESTION: {user_query}

PRIMARY MEDICATION: {drug_name}

FDA LABEL INFORMATION (Read ALL snippets before answering):
{context}

---

INSTRUCTIONS:"""

    if query_type == "boxed_warning":
        instructions = """
The user is asking specifically about the BOXED WARNING.

Your answer MUST:
1. Quote or paraphrase the exact boxed warning if present in the snippets
2. Explain what the warning means in plain English
3. Describe the serious risk or condition being warned about
4. If no boxed warning exists in the snippets, clearly state "This medication does not have a boxed warning in the provided label information."

Boxed warnings are the FDA's strongest safety warnings - treat this seriously."""

    elif query_type == "general" or query_type == "warnings":
        instructions = """
The user is asking WHY something is done or about warnings/safety.

Your answer MUST:
1. Start with the direct medical reason (therapeutic rationale)
2. Explain the mechanism if it's in the snippets
3. Include any relevant warnings or contraindications related to their question
4. Be thorough - this is clearly an important question

Read ALL snippets carefully. Look for:
• Indications (what it treats and why)
• Mechanism of action
• Warnings related to the scenario they asked about
• Clinical study results if relevant

Be comprehensive. Don't say "the label doesn't explain" if any snippet actually does explain it."""

    elif query_type == "interactions":
        instructions = """
The user is asking about drug interactions or using medications together.

Your answer MUST:
1. State whether they're meant to be used together (if indicated)
2. Explain the therapeutic reason if they're combined
3. List any warnings about how to administer them
4. Mention any risks or contraindications

Look across ALL snippets for:
• Indications mentioning combination therapy
• Drug interactions section
• Administration warnings
• Contraindications

Be complete - cover both the therapeutic use AND safety considerations."""

    elif query_type == "usage":
        instructions = """
Answer what the medication is used for.

Your answer MUST include:
1. All indicated uses from the label
2. Key warnings or contraindications that limit its use
3. Any boxed warnings (critical)

Check ALL snippets - don't just read the first one."""

    elif query_type == "side_effects":
        instructions = """
Explain the side effects from the label.

FORMAT:
• Use bullet points.
• Keep the answer within 5–6 lines.

CONTENT RULES:
• List the side effects exactly as found in the snippets.
• Highlight serious or urgent ones separately.
• Mention when someone should seek medical attention if the label specifies it."""

    elif query_type == "dosage":
        instructions = """
Answer the dosage question.

Your answer MUST:
1. Provide the specific dosing information from the label
2. Include any special instructions (with food, timing, etc.)
3. Mention what to do if a dose is missed (if provided)
4. Include relevant warnings about dosing

Be precise with numbers and frequencies."""

    elif query_type == "pregnancy":
        instructions = """
Answer the pregnancy/breastfeeding question.

Your answer MUST:
1. State the pregnancy category or risk if mentioned
2. Describe any known risks or warnings
3. Include breastfeeding information if relevant
4. Mention any contraindications

This is critical safety information - be thorough."""

    else:
        instructions = """
Provide a complete but still concise answer based strictly on the snippets.

FORMAT:
• Aim for about 6–10 lines.
• Start by directly answering the user's question in 1–2 lines.
• Then add 3–6 additional lines that summarize closely related directions, warnings, or safety advice from the snippets.
• Use bullet points for lists; a compact paragraph for explanations.

CONTENT RULES:
• Always restate the specific instruction the user asked about.
• Add brief context from the same label snippet (for example: how often to apply, general sun protection measures, or important warnings).
• Include any major safety warnings if relevant.
• Do not invent information not present in the label."""

    return base + instructions

def initialize_rag_system():
    global _metas, _drug_index, _index, _tokenizer, _model, _reranker, _gemini_model, _initialized

    if _initialized:
        return "System already initialized."

    print("Loading system components...")

    try:
        _metas = rag.load_metadata(META_FILE)
        _drug_index = rag.build_drug_index(_metas)
        _index = rag.load_index(INDEX_FILE)
        _tokenizer, _model = rag.load_model(MODEL_NAME)
        reranker_tok, reranker_model = rag.load_reranker(RERANKER_NAME)
        _reranker = (reranker_tok, reranker_model)
        _gemini_model = init_gemini()

        _initialized = True
        print("✓ System ready!\n")
        return "System initialized successfully."

    except Exception as e:
        error_msg = f"Initialization failed: {str(e)}"
        print(f"✗ {error_msg}")
        return error_msg

def rag_answer(
        user_query: str,
        few_shot_examples: str = "",
) -> str:
    global _metas, _drug_index, _index, _tokenizer, _model, _reranker, _gemini_model, _initialized

    if not _initialized:
        init_status = initialize_rag_system()
        if "failed" in init_status.lower():
            return f"System initialization error:\n{init_status}"

    hits = rag.search(
        query=user_query,
        index=_index,
        metas=_metas,
        tokenizer=_tokenizer,
        model=_model,
        reranker=_reranker,
        top_k=TOP_K,
        faiss_k=FAISS_K,
        drug_index=_drug_index,
    )

    if not hits:
        return (
            "I couldn't find information about that in the available drug labels. "
            "You might want to check the medication's package insert or ask your pharmacist."
        )

    prompt = build_adaptive_prompt(user_query, hits, few_shot_examples)

    try:
        response = _gemini_model.generate_content(prompt)
        answer = response.text.strip()
    except Exception as e:
        return f"Sorry, I had trouble generating a response: {str(e)}"

    return answer

def get_rag_response(user_query: str, few_shot_examples: str = "") -> str:
    """Streamlit-compatible wrapper for rag_answer."""
    return rag_answer(user_query, few_shot_examples)

def main():
    print("=" * 80)
    print("Comprehensive Pharmacy Assistant (RAG + Gemini)")
    print("=" * 80 + "\n")

    init_status = initialize_rag_system()

    print("Ask questions about medications. Type 'quit' to exit.\n")

    while True:
        q = input("Your question (or 'quit'): ").strip()
        if not q or q.lower() in ["quit", "exit", "q"]:
            break

        answer = rag_answer(user_query=q)

        print("\n" + answer)
        print("\n" + "-" * 80 + "\n")


if __name__ == "__main__":
    main()