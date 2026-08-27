import os
import io
import streamlit as st
from pathlib import Path
from dotenv import load_dotenv

# Ensure documind-nlp path is in sys.path
import sys
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from src.parsers import parse_document
from src.summarizer import generate_extractive_summary, generate_abstractive_summary
from src.entities import extract_entities, extract_keywords
from src.analytics import compute_document_analytics
from src.rag import split_document_pages, VectorStore, RAGEngine
from src.exporters import generate_docx_report, generate_pdf_report
from src.evaluation import evaluate_summary
from src.config import is_gemini_configured, GEMINI_MODEL

st.set_page_config(
    page_title="DocuMind AI — Smart NLP Platform",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #475569;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 0.75rem;
        padding: 1rem;
        text-align: center;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        padding: 8px 16px;
        border-radius: 6px;
    }
</style>
""", unsafe_allow_html=True)

# Session state initialization
if "doc_data" not in st.session_state:
    st.session_state.doc_data = None
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None
if "extractive_summary" not in st.session_state:
    st.session_state.extractive_summary = None
if "abstractive_summary" not in st.session_state:
    st.session_state.abstractive_summary = None
if "entities_data" not in st.session_state:
    st.session_state.entities_data = None
if "keywords_data" not in st.session_state:
    st.session_state.keywords_data = None
if "analytics_data" not in st.session_state:
    st.session_state.analytics_data = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "qa_history" not in st.session_state:
    st.session_state.qa_history = []

# --- SIDEBAR ---
with st.sidebar:
    st.image("https://raw.githubusercontent.com/lucide-icons/lucide/main/icons/brain.svg", width=48)
    st.title("DocuMind AI")
    st.caption("Smart NLP Platform — KOI")
    
    st.divider()
    
    # API Key Configuration
    gemini_key_input = st.text_input(
        "Gemini API Key",
        value=os.getenv("GEMINI_API_KEY", ""),
        type="password",
        help="Get a free key from https://aistudio.google.com/. Required for neural abstractive summaries and grounded Q&A."
    )
    if gemini_key_input:
        os.environ["GEMINI_API_KEY"] = gemini_key_input
        
    if is_gemini_configured():
        st.success(f"🟢 Gemini Ready ({GEMINI_MODEL})", icon="✅")
    else:
        st.info("💡 Running in Local Offline Mode (LexRank, NER, Keywords, and FAISS Vector Search are ready).", icon="ℹ️")

    st.divider()

    # File Upload
    uploaded_file = st.file_uploader(
        "Upload Document",
        type=["pdf", "docx", "txt"],
        help="Supports PDF, Word (DOCX), and Plain Text (TXT) files up to 20 MB."
    )

    if uploaded_file and (st.session_state.doc_data is None or st.session_state.doc_data.get("filename") != uploaded_file.name):
        with st.spinner("Parsing and indexing document..."):
            file_bytes = uploaded_file.getvalue()
            doc_data = parse_document(file_bytes, uploaded_file.name)
            st.session_state.doc_data = doc_data
            
            # Auto-compute analytics
            st.session_state.analytics_data = compute_document_analytics(doc_data["cleaned_text"], doc_data["filename"])
            
            # Chunk and build FAISS index
            chunks = split_document_pages(doc_data["pages"])
            vs = VectorStore()
            vs.build_index(chunks)
            st.session_state.vector_store = vs
            
            # Reset pipeline states for new doc
            st.session_state.extractive_summary = None
            st.session_state.abstractive_summary = None
            st.session_state.entities_data = None
            st.session_state.keywords_data = None
            st.session_state.chat_history = []
            st.session_state.qa_history = []
            st.success(f"Loaded: {uploaded_file.name} ({doc_data['page_count']} pages)")

    st.divider()
    st.markdown("### Architecture Pipeline")
    st.caption("• **Parser**: `pdfplumber` / `pypdf` / `python-docx`\n• **Embeddings**: `all-MiniLM-L6-v2` (Local CPU)\n• **Summaries**: LexRank + Gemini Flash\n• **NER/Keywords**: `spaCy` + `KeyBERT`\n• **Exporters**: `python-docx` + `reportlab`")

# --- MAIN CONTENT ---
st.markdown('<div class="main-header">Smart NLP Document Intelligence Platform</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Automated Ingestion, Dual Summarisation, Grounded RAG Q&A, and Multi-Format Reports</div>', unsafe_allow_html=True)

if not st.session_state.doc_data:
    st.info("👈 Please upload a `.pdf`, `.docx`, or `.txt` document in the sidebar to begin analysis.")
    st.stop()

doc = st.session_state.doc_data
analytics = st.session_state.analytics_data

# Top Level KPI Row
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("File Name", doc["filename"])
c2.metric("Page Count", doc["page_count"])
c3.metric("Word Count", f"{analytics['word_count']:,}")
c4.metric("Reading Time", f"{analytics['reading_time_min']} min")
c5.metric("Readability", f"{analytics['readability']['score']} / 100", help=f"{analytics['readability']['level']} ({analytics['readability']['grade']})")

# Tabs
tab_analytics, tab_summary, tab_qa, tab_entities, tab_export = st.tabs([
    "📄 Document & Analytics",
    "📝 Dual Summaries",
    "💬 Grounded Q&A (RAG)",
    "🏷️ Entities & Topics",
    "📊 Evaluation & Export"
])

# --- TAB 1: DOCUMENT & ANALYTICS ---
with tab_analytics:
    st.subheader("Document Metrics & Structure")
    
    col_stat1, col_stat2 = st.columns(2)
    with col_stat1:
        st.markdown("**Reading & Structure Statistics**")
        st.write(f"- **Total Characters:** {analytics['char_count']:,}")
        st.write(f"- **Total Sentences:** {analytics['sentence_count']:,}")
        st.write(f"- **Paragraphs:** {analytics['paragraph_count']:,}")
        st.write(f"- **Vocabulary Richness (TTR):** {analytics['vocabulary_richness_pct']}% ({analytics['unique_words']:,} unique words)")
        st.write(f"- **Avg Words per Sentence:** {analytics['avg_words_per_sentence']}")
        st.write(f"- **Estimated Speaking Time:** {analytics['speaking_time_min']} min")
        
    with col_stat2:
        st.markdown("**Readability Score (Flesch Formula)**")
        st.info(f"**Level:** {analytics['readability']['level']}\n\n**Target Audience:** {analytics['readability']['grade']}")
        
    st.divider()
    st.subheader("Extracted Text by Page")
    selected_page = st.selectbox("Select Page to View", options=[p["page_number"] for p in doc["pages"]])
    page_content = next((p for p in doc["pages"] if p["page_number"] == selected_page), None)
    if page_content:
        st.text_area(f"Page {selected_page} Content ({page_content['word_count']} words)", page_content["cleaned_text"], height=300)

# --- TAB 2: DUAL SUMMARIES ---
with tab_summary:
    st.subheader("Dual Summarisation Engine")
    st.caption("Extractive sentence ranking alongside neural abstractive synthesis for direct verification.")
    
    ctrl_col1, ctrl_col2, ctrl_col3 = st.columns([1.5, 1.5, 1])
    with ctrl_col1:
        num_sentences = st.slider("Extractive Sentences (LexRank)", min_value=3, max_value=12, value=5)
    with ctrl_col2:
        abs_length = st.selectbox("Abstractive Length", options=["brief", "standard", "detailed"], index=1)
        abs_format = st.selectbox("Abstractive Format", options=["paragraph", "bullet_points", "executive"], index=0)
    with ctrl_col3:
        focus_topic = st.text_input("Focus Keyword (Optional)", placeholder="e.g. Methodology")
        
    if st.button("⚡ Generate Both Summaries", type="primary"):
        with st.spinner("Generating extractive & neural abstractive summaries..."):
            st.session_state.extractive_summary = generate_extractive_summary(doc["cleaned_text"], sentence_count=num_sentences)
            st.session_state.abstractive_summary = generate_abstractive_summary(
                doc["cleaned_text"], format_type=abs_format, max_length=abs_length, focus_topic=focus_topic
            )
            
    col_ext, col_abs = st.columns(2)
    with col_ext:
        st.markdown("#### 1. Extractive Summary (LexRank)")
        st.caption("Top-ranked original sentences from source text (100% faithful, zero hallucination).")
        if st.session_state.extractive_summary:
            for s in st.session_state.extractive_summary["sentences"]:
                st.markdown(f"• {s}")
        else:
            st.info("Click 'Generate Both Summaries' above.")
            
    with col_abs:
        st.markdown("#### 2. Abstractive Neural Summary (Gemini Flash)")
        st.caption("Cohesive neural synthesis summarizing core themes and conclusions.")
        if st.session_state.abstractive_summary:
            st.markdown(st.session_state.abstractive_summary["summary"])
        else:
            st.info("Click 'Generate Both Summaries' above.")

# --- TAB 3: GROUNDED Q&A (RAG) ---
with tab_qa:
    st.subheader("Grounded Question Answering (RAG)")
    st.caption("Query document contents. Every answer is grounded strictly in retrieved passages with page citations.")

    user_query = st.chat_input("Ask a question about this document...")
    
    # Display chat history
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("sources"):
                with st.expander("🔍 View Retrieved Source Passages"):
                    for src in msg["sources"]:
                        st.markdown(f"**[Page {src['page']} | Score: {src['similarity_score']}]**\n> {src['snippet']}")

    if user_query:
        # User message
        st.session_state.chat_history.append({"role": "user", "content": user_query})
        with st.chat_message("user"):
            st.markdown(user_query)

        # Assistant response
        with st.chat_message("assistant"):
            with st.spinner("Retrieving relevant passages and synthesizing answer..."):
                rag_engine = RAGEngine(st.session_state.vector_store)
                res = rag_engine.answer_question(
                    user_query,
                    top_k=4,
                    chat_history=st.session_state.chat_history[:-1]
                )
                st.markdown(res["answer"])
                if res["sources"]:
                    with st.expander("🔍 View Retrieved Source Passages"):
                        for src in res["sources"]:
                            st.markdown(f"**[Page {src['page']} | Score: {src['similarity_score']}]**\n> {src['snippet']}")

                # Save assistant response
                st.session_state.chat_history.append({
                    "role": "assistant",
                    "content": res["answer"],
                    "sources": res["sources"]
                })
                # Add to persistent QA export history
                st.session_state.qa_history.append({
                    "question": user_query,
                    "answer": res["answer"],
                    "sources": res["sources"]
                })

# --- TAB 4: ENTITIES & TOPICS ---
with tab_entities:
    st.subheader("Named Entity Recognition & High-Value Key Phrases")
    st.caption("Extracted using `spaCy` NER and `KeyBERT` semantic topic ranking.")
    
    if st.button("🏷️ Extract Entities & Keywords", type="primary") or not st.session_state.entities_data:
        with st.spinner("Extracting entities and keyphrases..."):
            st.session_state.entities_data = extract_entities(doc["cleaned_text"])
            st.session_state.keywords_data = extract_keywords(doc["cleaned_text"])

    col_kw, col_ent = st.columns([1, 1.5])
    with col_kw:
        st.markdown("#### Top Key Phrases (KeyBERT)")
        if st.session_state.keywords_data:
            for kw in st.session_state.keywords_data:
                st.markdown(f"- **{kw['keyword']}** `(score: {kw['score']})`")
                
    with col_ent:
        st.markdown("#### Named Entities (spaCy)")
        if st.session_state.entities_data:
            for group, items in st.session_state.entities_data["entities_by_type"].items():
                if items:
                    st.markdown(f"**{group}**")
                    badges = " ".join([f"`{it['name']} ({it['count']})`" for it in items])
                    st.markdown(badges)
                    st.write("")

# --- TAB 5: EVALUATION & EXPORT ---
with tab_export:
    st.subheader("1. Report Export Engine")
    st.caption("Export your complete document analysis (metadata, summaries, entities, and Q&A history) into formatted reports.")
    
    col_exp1, col_exp2 = st.columns(2)
    with col_exp1:
        if st.button("📄 Generate DOCX Report (.docx)", use_container_width=True):
            docx_data = generate_docx_report(
                document_info=doc,
                extractive_summary=st.session_state.extractive_summary,
                abstractive_summary=st.session_state.abstractive_summary,
                entities_data=st.session_state.entities_data,
                keywords_data=st.session_state.keywords_data,
                analytics_data=st.session_state.analytics_data,
                qa_history=st.session_state.qa_history
            )
            st.download_button(
                label="⬇️ Download DOCX Report",
                data=docx_data,
                file_name=f"DocuMind_Report_{Path(doc['filename']).stem}.docx",
                mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                use_container_width=True
            )
            
    with col_exp2:
        if st.button("📑 Generate PDF Report (.pdf)", use_container_width=True):
            pdf_data = generate_pdf_report(
                document_info=doc,
                extractive_summary=st.session_state.extractive_summary,
                abstractive_summary=st.session_state.abstractive_summary,
                entities_data=st.session_state.entities_data,
                keywords_data=st.session_state.keywords_data,
                analytics_data=st.session_state.analytics_data,
                qa_history=st.session_state.qa_history
            )
            st.download_button(
                label="⬇️ Download PDF Report",
                data=pdf_data,
                file_name=f"DocuMind_Report_{Path(doc['filename']).stem}.pdf",
                mime="application/pdf",
                use_container_width=True
            )

    st.divider()
    st.subheader("2. Quantitative Evaluation Suite (ROUGE & BLEU)")
    st.caption("Benchmark candidate summary quality against reference ground-truth text.")
    
    default_cand = st.session_state.abstractive_summary["summary"] if st.session_state.abstractive_summary else (
        st.session_state.extractive_summary["summary"] if st.session_state.extractive_summary else ""
    )
    
    col_eval1, col_eval2 = st.columns(2)
    with col_eval1:
        ref_text = st.text_area("Reference Summary (Ground Truth)", height=150, placeholder="Paste expected reference summary here...")
    with col_eval2:
        cand_text = st.text_area("Candidate Summary to Evaluate", value=default_cand, height=150)
        
    if st.button("📊 Calculate ROUGE & BLEU Scores"):
        if not ref_text.strip() or not cand_text.strip():
            st.warning("Please provide both reference and candidate text.")
        else:
            scores = evaluate_summary(ref_text, cand_text)
            e1, e2, e3, e4 = st.columns(4)
            e1.metric("ROUGE-1 (F1)", f"{scores['rouge1']['fmeasure']}%", f"P: {scores['rouge1']['precision']}% | R: {scores['rouge1']['recall']}%")
            e2.metric("ROUGE-2 (F1)", f"{scores['rouge2']['fmeasure']}%", f"P: {scores['rouge2']['precision']}% | R: {scores['rouge2']['recall']}%")
            e3.metric("ROUGE-L (F1)", f"{scores['rougeL']['fmeasure']}%", f"P: {scores['rougeL']['precision']}% | R: {scores['rougeL']['recall']}%")
            e4.metric("SacreBLEU", f"{scores['bleu']['score']}", help="Corpus BLEU score (0-100 scale)")
