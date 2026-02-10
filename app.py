import streamlit as st
from rag import ingest_documents, retrieve_answer, llm
from tools import finance_advisor_tool, yfinance_tool, web_search_full
import logging
from datetime import datetime
import re
try:
    import sympy as sp
except Exception:
    sp = None

# Setup Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),  # Console output
        logging.FileHandler('rag_app.log', encoding='utf-8')  # File with UTF-8
    ]
)
logger = logging.getLogger(__name__)
logger.info("=" * 80)
logger.info("[START] RAG Application Started")
logger.info("=" * 80)

st.set_page_config(page_title="FINWISE", layout="wide")

# Session state initialization
if "web_permission" not in st.session_state: st.session_state.web_permission = None
if "pending_query" not in st.session_state: st.session_state.pending_query = None
if "uploaded_files" not in st.session_state: st.session_state.uploaded_files = []
if "force_web" not in st.session_state: st.session_state.force_web = False

st.title("🏦 FINWISE")

with st.sidebar:
    st.header("[FILE] Document Management")
    logger.info("[FOLDER] Document sidebar opened")
    
    # File uploader for txt and pdf
    uploaded = st.file_uploader(
        "Upload files (PDF or TXT)", 
        accept_multiple_files=True,
        type=["pdf", "txt"]
    )
    
    if uploaded:
        logger.info(f"[UPLOAD] User uploaded {len(uploaded)} file(s)")

        # Prepare lists
        texts = []
        successful_uploads = []
        failed_uploads = []

        # Show simple filename list immediately
        st.write("**Upload Status:**")
        for file in uploaded:
            st.write(f"- {file.name}")

        # Process files (no per-file success badges)
        for file in uploaded:
            try:
                logger.info(f"[FILE] Processing file: {file.name} (Type: {file.type})")
                if file.type == "application/pdf":
                    logger.info(f"[SEARCH] Extracting text from PDF: {file.name}")
                    from PyPDF2 import PdfReader
                    pdf_reader = PdfReader(file)
                    text = " ".join([page.extract_text() for page in pdf_reader.pages])
                    texts.append(text)
                    successful_uploads.append(file.name)
                elif file.type == "text/plain":
                    logger.info(f"[SEARCH] Reading text file: {file.name}")
                    text = file.read().decode('utf-8')
                    texts.append(text)
                    successful_uploads.append(file.name)
            except Exception as e:
                logger.error(f"[ERROR] Error processing {file.name}: {str(e)}")
                failed_uploads.append((file.name, str(e)))

        # Ingest successfully uploaded documents
        if texts:
            logger.info(f"[PROCESS] Ingesting {len(texts)} document(s) into vector store...")
            with st.spinner(f"Indexing {len(successful_uploads)} file(s)..."):
                added = ingest_documents(texts)
                if added and added > 0:
                    logger.info(f"[OK] Successfully indexed {len(successful_uploads)} file(s) (added {added} embeddings)")
                    # Only keep file names
                    st.session_state.uploaded_files.extend(successful_uploads)
                    st.success(f"[OK] Indexed {len(successful_uploads)} file(s) successfully!")
                else:
                    logger.warning("[WARN] Indexing returned no new embeddings")

        if failed_uploads:
            logger.warning(f"[WARN] Failed to upload {len(failed_uploads)} file(s)")

        # Show stored filenames
        if st.session_state.uploaded_files:
            st.divider()
            st.write("**[DOCS] Stored Files:**")
            for stored_file in st.session_state.uploaded_files:
                st.write(f"  • {stored_file}")

# Intent Logic
def get_intent(q):
    q = q.lower()
    if any(k in q for k in ['price', 'ticker', '$']): return "price"
    if any(k in q for k in ['should i', 'invest', 'advice', 'plan']): return "advisor"
    return "general"

def is_math_query(s: str) -> bool:
    s = s.strip()
    if not s:
        return False
    math_keywords = ['solve', 'integrate', 'differentiate', 'derivative', 'integral', 'limit']
    if any(k in s.lower() for k in math_keywords):
        return True
    if re.fullmatch(r"[0-9\s\+\-\*\/\^\.(\)x=]+", s.replace('**','^')):
        return True
    return False

# Track last query to avoid re-processing on keystrokes
if "last_query" not in st.session_state:
    st.session_state.last_query = ""

query = st.text_input("Ask a question:")

# Process query only if it's new OR if force_web is active
if st.session_state.force_web and st.session_state.web_permission == 'yes':
    # Continue forced web search - pending_query already set
    pass
elif query and query != st.session_state.last_query:
    logger.info(f"[Q] User query received: {query}")
    st.session_state.pending_query = query
    st.session_state.web_permission = None
    st.session_state.force_web = False
    st.session_state.last_query = query

if st.session_state.pending_query:
    query = st.session_state.pending_query
    # Math handling
    if is_math_query(query):
        logger.info("[MATH] Detected math query")
        if sp is None:
            st.error("Math support requires 'sympy' package. See requirements.txt")
        else:
            try:
                if '=' in query:
                    lhs, rhs = query.split('=', 1)
                    expr_l = sp.sympify(lhs)
                    expr_r = sp.sympify(rhs)
                    sol = sp.solve(sp.Eq(expr_l, expr_r))
                    st.subheader('Math Solution')
                    st.write('Equation:', lhs, '=', rhs)
                    st.write('Solution:', sol)
                else:
                    expr = sp.sympify(query)
                    simplified = sp.simplify(expr)
                    evaluated = None
                    try:
                        evaluated = float(sp.N(expr))
                    except Exception:
                        evaluated = None
                    st.subheader('Math Result')
                    st.write('Original:', query)
                    st.write('Simplified:', str(simplified))
                    if evaluated is not None:
                        st.write('Numeric evaluation:', evaluated)
            except Exception as e:
                logger.error(f"[MATH_ERR] {e}")
                st.error(f"Error solving math query: {e}")
        st.session_state.pending_query = None
        st.session_state.force_web = False
    else:
        intent = get_intent(query)
        logger.info(f"[AIM] Query intent detected: {intent}")

        # Force web path
        if st.session_state.force_web and st.session_state.web_permission == 'yes':
            logger.info("[WEB] Force web search active - performing web search")
            with st.spinner("Searching the Web..."):
                ans = web_search_full(st.session_state.pending_query, llm)
                st.subheader('Web Answer')
                st.write(ans)
            st.session_state.pending_query = None
            st.session_state.web_permission = None
            st.session_state.force_web = False
        else:
            if st.session_state.web_permission is None:
                logger.info("[DOCS] Searching in document database...")
                with st.spinner('Searching Documents...'):
                    doc_ans = retrieve_answer(query)
                logger.info(f"[FILE] Document search result: {doc_ans[:100]}..." if len(doc_ans) > 100 else f"[FILE] Document search result: {doc_ans}")

                if "NOT_FOUND" not in doc_ans:
                    logger.info("[OK] Answer found in documents")
                    st.subheader('Document Answer')
                    st.write(doc_ans)
                    st.session_state.pending_query = None
                else:
                    logger.info("[ERROR] Answer NOT found in documents - asking for web search permission")
                    st.warning('Answer not found in documents.')
                    st.info('Would you like to search the web for an answer?')
                    col1, col2 = st.columns(2)
                    with col1:
                        if st.button('Yes, Search Web', key='btn_yes'):
                            logger.info('[OK] User granted web search permission - YES (force web)')
                            st.session_state.web_permission = 'yes'
                            st.session_state.force_web = True
                            st.rerun()
                    with col2:
                        if st.button('No, Thanks', key='btn_no'):
                            logger.info('[ERROR] User declined web search permission - NO')
                            st.session_state.web_permission = 'no'
                            st.session_state.force_web = False
                            st.rerun()
            else:
                if st.session_state.web_permission == 'yes':
                    logger.info('[WEB] Web search permission confirmed - proceeding with web search')
                    with st.spinner('Searching the Web...'):
                        ans = web_search_full(st.session_state.pending_query, llm)
                        st.subheader('Web Answer')
                        st.write(ans)
                    st.session_state.web_permission = None
                    st.session_state.pending_query = None
                    st.session_state.force_web = False
                elif st.session_state.web_permission == 'no':
                    logger.info('[BLOCKED] Web search cancelled by user')
                    st.error("Search cancelled. Answer stays 'Not Found' in local documents.")
                    st.session_state.web_permission = None
                    st.session_state.pending_query = None