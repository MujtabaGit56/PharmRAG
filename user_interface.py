import streamlit as st
import llm_inference
import retrieval_from_rag as rag
from drug_detection import detect_drug_from_query_robust
import time

st.set_page_config(page_title="Pharmacy RAG Assistant", layout="wide", initial_sidebar_state="collapsed")

st.markdown("""
<style>
    .stTextArea textarea {
        font-family: monospace;
    }

    /* Hide the "press enter" hint */
    .stTextInput > label > div[data-testid="stMarkdownContainer"] > p {
        display: none;
    }

    /* Remove extra spacing and form messages */
    .stForm {
        border: none !important;
        padding: 0 !important;
    }

    .stTextInput > div > div > input {
        caret-color: auto;
    }

    /* Drug detection badge */
    .drug-badge {
        display: inline-block;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 8px 16px;
        border-radius: 20px;
        font-weight: 600;
        font-size: 14px;
        margin: 10px 0;
        box-shadow: 0 2px 8px rgba(102, 126, 234, 0.3);
    }

    /* Info card */
    .info-card {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        padding: 20px;
        border-radius: 12px;
        margin: 15px 0;
        border-left: 4px solid #667eea;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
    }

    /* Response card */
    .response-card {
        background: white;
        padding: 25px;
        border-radius: 12px;
        border-left: 4px solid #10b981;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        margin: 15px 0;
    }

    /* Metadata section */
    .metadata {
        background: #f9fafb;
        padding: 15px;
        border-radius: 8px;
        margin-top: 15px;
        font-size: 0.9em;
        color: #6b7280;
    }

    /* Section headers */
    .section-header {
        font-size: 1.1em;
        font-weight: 600;
        color: #374151;
        margin: 20px 0 10px 0;
        display: flex;
        align-items: center;
        gap: 8px;
    }

    /* Reset button styling */
    .stButton > button[kind="secondary"] {
        background-color: #f3f4f6;
        color: #374151;
        border: 1px solid #d1d5db;
    }

    .stButton > button[kind="secondary"]:hover {
        background-color: #e5e7eb;
        border-color: #9ca3af;
    }
</style>
""", unsafe_allow_html=True)

if 'initialized' not in st.session_state:
    st.session_state.initialized = False
    st.session_state.system_ready = False
    st.session_state.show_response = False
    st.session_state.current_query = ""
    st.session_state.last_response = ""
    st.session_state.last_metadata = {}

if not st.session_state.initialized:
    with st.spinner("🔧 Initializing RAG system..."):
        try:
            llm_inference.initialize_rag_system()
            st.session_state.system_ready = True
            st.session_state.initialized = True
        except Exception as e:
            st.error(f"Failed to initialize system: {e}")
            st.session_state.system_ready = False

st.title(" Pharmacy Research Assistant")
st.markdown("*Powered by FDA label information*")
st.markdown("---")

col_left, col_right = st.columns([1, 2], gap="large")

with col_left:
    st.subheader(" Configuration")

    user_query = st.text_input(
        "Enter your query:",
        value=st.session_state.current_query if st.session_state.show_response else "",
        placeholder="e.g., What are the contraindications for Amoxicillin?",
        help="Type your pharmacy-related question here.",
        key="query_input",
        label_visibility="visible"
    )

    btn_col1, btn_col2 = st.columns([2, 1])
    with btn_col1:
        submit_button = st.button("🔍 Generate Response", type="primary", use_container_width=True)
    with btn_col2:
        reset_button = st.button("🔄 Reset", type="secondary", use_container_width=True)

    st.write(" ")

    with st.expander("Few-Shot Examples (Optional)", expanded=False):
        default_examples = """Q: According to the provided label, what is the exact recommended frequency for applying Blue Ice Analgesic to the affected area for adults and children 2 years of age and older?
A: For adults and children 2 years of age and older, apply Blue Ice Analgesic to the affected area not more than 3 to 4 times daily. This product is for external use only; avoid contact with eyes. Do not apply to wounds or damaged skin, and do not bandage tightly. Stop use and ask a doctor if the condition worsens, symptoms persist for more than 7 days, or clear up and occur again within a few days. For children under 2 years of age, consult a doctor before use.

Q: Does the label for Amlodipine Besylate explicitly mention that doses in excess of 5 mg daily have not been studied in pediatric patients?
A: Yes, the label for Amlodipine Besylate explicitly states that doses in excess of 5 mg daily have not been studied in pediatric patients. The recommended pediatric starting dose is 2.5 mg to 5 mg once daily. This is noted as an important limitation for pediatric patients. For adults, the usual initial dose is 5 mg once daily, with a maximum dose of 10 mg once daily."""

        few_shot_examples = st.text_area(
            "Edit examples to guide model behavior:",
            value=default_examples,
            height=250,
            help="These examples help the model understand the desired response format."
        )

if reset_button:
    st.session_state.show_response = False
    st.session_state.current_query = ""
    st.session_state.last_response = ""
    st.session_state.last_metadata = {}
    st.rerun()

with col_right:
    st.subheader(" Output")

    if not st.session_state.system_ready:
        st.error("⚠️ System not initialized. Please refresh the page.")
    elif submit_button and user_query:
        st.session_state.current_query = user_query
        st.session_state.show_response = True
        status_placeholder = st.empty()
        drug_placeholder = st.empty()
        retrieval_placeholder = st.empty()
        response_placeholder = st.empty()
        metadata_placeholder = st.empty()

        try:
            status_placeholder.info("🔍 Analyzing query...")
            time.sleep(0.3)

            detected_drug = None
            if llm_inference._drug_index:
                detected_drug = detect_drug_from_query_robust(
                    user_query,
                    llm_inference._drug_index,
                    verbose=False
                )

            if detected_drug:
                drug_placeholder.markdown(f"""
                <div class="info-card">
                    <div class="section-header">🎯 Drug Detected</div>
                    <div class="drug-badge">{detected_drug.upper()}</div>
                    <p style="margin-top: 10px; color: #6b7280;">
                        The system has identified this medication in your query and will prioritize 
                        relevant FDA label information.
                    </p>
                </div>
                """, unsafe_allow_html=True)
            else:
                drug_placeholder.markdown("""
                <div class="info-card">
                    <div class="section-header">🔍 Analysis</div>
                    <p style="color: #6b7280;">
                        No specific drug detected. Searching across all available FDA labels...
                    </p>
                </div>
                """, unsafe_allow_html=True)

            status_placeholder.info("📚 Retrieving relevant FDA label sections...")
            time.sleep(0.5)

            hits = rag.search(
                query=user_query,
                index=llm_inference._index,
                metas=llm_inference._metas,
                tokenizer=llm_inference._tokenizer,
                model=llm_inference._model,
                reranker=llm_inference._reranker,
                top_k=10,
                faiss_k=250,
                drug_index=llm_inference._drug_index
            )

            if hits:
                top_sources = set()
                for score, meta in hits[:3]:
                    drug_name = meta.get('drug_name', 'Unknown')
                    section = meta.get('section', 'Unknown')
                    top_sources.add(f"{drug_name} - {section}")

                retrieval_placeholder.markdown(f"""
                <div class="info-card">
                    <div class="section-header">📄 Sources Retrieved</div>
                    <p style="color: #6b7280; margin-bottom: 8px;">
                        Found {len(hits)} relevant sections. Top sources:
                    </p>
                    <ul style="color: #374151; margin-left: 20px;">
                        {"".join([f"<li>{source}</li>" for source in list(top_sources)[:3]])}
                    </ul>
                </div>
                """, unsafe_allow_html=True)

            status_placeholder.info("🤖 Generating response from AI model...")
            time.sleep(0.3)

            result = llm_inference.get_rag_response(
                user_query,
                few_shot_examples if 'few_shot_examples' in locals() else ""
            )

            st.session_state.last_response = result
            st.session_state.last_metadata = {
                'query_type': rag.classify_query_intent(user_query),
                'num_hits': len(hits),
                'detected_drug': detected_drug
            }

            status_placeholder.success("✅ Response generated successfully!")
            time.sleep(0.5)
            status_placeholder.empty()

            response_placeholder.markdown(f"""
            <div class="response-card">
                <div class="section-header">💬 AI Response</div>
                {result}
            </div>
            """, unsafe_allow_html=True)

            metadata_placeholder.markdown(f"""
            <div class="metadata">
                <div style="display: flex; justify-content: space-between; flex-wrap: wrap; gap: 15px;">
                    <div>
                        <strong>Query Type:</strong> {rag.classify_query_intent(user_query).replace('_', ' ').title()}
                    </div>
                    <div>
                        <strong>Sources Used:</strong> {len(hits)} FDA label sections
                    </div>
                </div>
                <div style="margin-top: 10px; padding-top: 10px; border-top: 1px solid #e5e7eb;">
                    <small>⚠️ This information is based on FDA labels and should not replace professional medical advice. 
                    Always consult with a healthcare provider for medical decisions.</small>
                </div>
            </div>
            """, unsafe_allow_html=True)

        except Exception as e:
            status_placeholder.empty()
            st.error(f"❌ An error occurred: {str(e)}")
            st.session_state.show_response = False

    elif submit_button and not user_query:
        st.warning("⚠️ Please enter a query first.")
    elif st.session_state.show_response and st.session_state.last_response:
        st.markdown(f"""
        <div class="response-card">
            <div class="section-header">💬 AI Response</div>
            {st.session_state.last_response}
        </div>
        """, unsafe_allow_html=True)
        meta = st.session_state.last_metadata
        st.markdown(f"""
        <div class="metadata">
            <div style="display: flex; justify-content: space-between; flex-wrap: wrap; gap: 15px;">
                <div>
                    <strong>Query Type:</strong> {meta.get('query_type', 'N/A').replace('_', ' ').title()}
                </div>
                <div>
                    <strong>Sources Used:</strong> {meta.get('num_hits', 0)} FDA label sections
                </div>
            </div>
            <div style="margin-top: 10px; padding-top: 10px; border-top: 1px solid #e5e7eb;">
                <small>⚠️ This information is based on FDA labels and should not replace professional medical advice. 
                Always consult with a healthcare provider for medical decisions.</small>
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.info("👋 Welcome! Enter a query and click 'Generate Response' to get started.")
        st.markdown("""
        <div style='background: linear-gradient(135deg, #667eea15 0%, #764ba215 100%); 
                    padding: 30px; border-radius: 12px; margin-top: 20px;
                    border: 2px dashed #667eea40;'>
            <div style='text-align: center; color: #6b7280;'>
                <h3 style='color: #374151; margin-bottom: 15px;'>💡 Example Queries</h3>
                <ul style='list-style: none; padding: 0; text-align: left; max-width: 500px; margin: 0 auto;'>
                    <li style='padding: 8px 0;'>• What are the side effects of ibuprofen?</li>
                    <li style='padding: 8px 0;'>• Can I take aspirin during pregnancy?</li>
                    <li style='padding: 8px 0;'>• What is the recommended dosage for amoxicillin?</li>
                    <li style='padding: 8px 0;'>• Are there any drug interactions with warfarin?</li>
                </ul>
            </div>
        </div>
        """, unsafe_allow_html=True)

with st.sidebar:
    st.header("ℹ️ About")
    st.markdown("""
    This assistant uses:
    - **Retrieval-Augmented Generation (RAG)** to find relevant FDA label information
    - **AI model** to generate clear, accurate responses
    - **Drug detection** to identify medications in your queries

    ### How it works:
    1.  Analyzes your query
    2.  Detects mentioned drugs
    3.  Retrieves relevant FDA sections
    4.  Generates a comprehensive answer

    ### Tips:
    - Be specific in your questions
    - Mention drug names when possible
    - Ask about one topic at a time
    - Use the Reset button to start fresh
    """)

    if st.session_state.system_ready:
        st.success("✅ System Ready")
    else:
        st.error("❌ System Not Ready")