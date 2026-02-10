import hashlib
import streamlit as st
import time
import logging
from langchain_ollama import OllamaEmbeddings, OllamaLLM
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

# Setup logger
logger = logging.getLogger(__name__)

CHROMA_PATH = "chroma_db"

@st.cache_resource
def get_vectorstore():
    logger.info("[LINK] Initializing vector store with Chroma")
    embeddings = OllamaEmbeddings(model="mxbai-embed-large")
    vectorstore = Chroma(persist_directory=CHROMA_PATH, embedding_function=embeddings)
    logger.info(f"[OK] Vector store initialized. Current documents: {vectorstore._collection.count()}")
    return vectorstore

llm = OllamaLLM(model="gemma3:4b", temperature=0)
logger.info("[BRAIN] LLM initialized: gemma3:4b")

def ingest_documents(texts):
    logger.info(f"[UPLOAD] Starting document ingestion for {len(texts)} text(s)")
    vectorstore = get_vectorstore()
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    
    # Track existing documents
    existing_docs = vectorstore.get()
    existing_ids = set(existing_docs['ids']) if existing_docs else set()
    logger.info(f"[CHART] Found {len(existing_ids)} existing document(s) in vector store")
    
    all_chunks = []

    # Progress bar UI
    progress_bar = st.progress(0, text="Splitting documents into chunks...")
    
    for i, text in enumerate(texts):
        file_hash = hashlib.md5(text.encode('utf-8')).hexdigest()
        logger.info(f"[SEARCH] Processing text {i+1}/{len(texts)}, hash: {file_hash}")
        
        # Check if document already exists
        if any(file_hash in doc_id for doc_id in existing_ids):
            logger.warning(f"[NEXT] Document {i+1} already exists in vector store, skipping")
            continue
        
        chunks = splitter.split_text(text)
        logger.info(f"[CUT] Split text {i+1} into {len(chunks)} chunks")
        
        for j, chunk in enumerate(chunks):
            all_chunks.append(Document(page_content=chunk, metadata={"file_id": file_hash}))
        
        # Update progress
        percent = int(((i + 1) / len(texts)) * 50)
        progress_bar.progress(percent, text=f"Processing file {i+1}/{len(texts)}...")

    if all_chunks:
        logger.info(f"[BRAIN] Embedding {len(all_chunks)} chunks into vector store")
        # Embedding progress
        batch_size = 5
        initial_count = vectorstore._collection.count()
        added_ids = []
        for i in range(0, len(all_chunks), batch_size):
            batch = all_chunks[i : i + batch_size]
            ids = [f"{doc.metadata['file_id']}_{idx}" for idx, doc in enumerate(batch, start=i)]
            vectorstore.add_documents(batch, ids=ids)
            added_ids.extend(ids)
            logger.info(f"[OK] Embedded batch {i//batch_size + 1}/{(len(all_chunks)-1)//batch_size + 1}")
            
            percent = 50 + int(((i + len(batch)) / len(all_chunks)) * 50)
            progress_bar.progress(percent, text=f"Embedding chunks {i+len(batch)}/{len(all_chunks)}...")
        
        time.sleep(1)
        progress_bar.empty()
        total_docs = vectorstore._collection.count()
        newly_added = total_docs - initial_count
        st.sidebar.info(f"[CHART] Total Saved Embeddings: {total_docs}")
        logger.info(f"[OK] Document ingestion complete. Total embeddings: {total_docs} (added {newly_added})")
        return newly_added
    
    logger.warning("[WARN] No new chunks to embed")
    return 0

def retrieve_answer(query: str) -> str:
    logger.info(f"[SEARCH] Retrieving answer for query: {query}")
    vectorstore = get_vectorstore()
    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
    docs = retriever.invoke(query)
    
    if not docs:
        logger.warning("[ERROR] No documents found for query")
        return "NOT_FOUND"
    
    logger.info(f"[DOCS] Found {len(docs)} relevant document(s)")
    for idx, doc in enumerate(docs, 1):
        logger.info(f"   {idx}. Relevance score for chunk: {doc.page_content[:50]}...")
    
    context = "\n\n".join([d.page_content for d in docs])
    prompt = f"Answer strictly using context: {context}\n\nIf missing, say 'NOT_FOUND'.\nQuestion: {query}"
    logger.info("[BRAIN] Generating answer from LLM")
    answer = llm.invoke(prompt).strip()
    logger.info(f"[OK] Answer generated: {answer[:100]}...")
    return answer