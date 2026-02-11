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

def ingest_documents(texts, file_names=None):
    logger.info(f"[UPLOAD] Starting document ingestion for {len(texts)} text(s)")
    vectorstore = get_vectorstore()
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    
    # Track existing documents
    existing_docs = vectorstore.get()
    existing_ids = set(existing_docs['ids']) if existing_docs else set()
    existing_metadatas = existing_docs['metadatas'] if existing_docs else []
    existing_hashes = {meta.get('file_hash'): meta.get('file_name', 'Unknown') for meta in existing_metadatas if meta}
    logger.info(f"[CHART] Found {len(existing_ids)} existing document(s) in vector store")
    
    all_chunks = []
    duplicate_files = []
    successfully_ingested = []
    
    if file_names is None:
        file_names = [f"Document_{i}" for i in range(len(texts))]
    
    for i, text in enumerate(texts):
        file_hash = hashlib.md5(text.encode('utf-8')).hexdigest()
        file_name = file_names[i] if i < len(file_names) else f"Document_{i}"
        logger.info(f"[SEARCH] Processing text {i+1}/{len(texts)}, hash: {file_hash}, name: {file_name}")
        
        # Check if document already exists
        if file_hash in existing_hashes:
            logger.warning(f"[NEXT] Document '{file_name}' already exists in vector store, skipping")
            duplicate_files.append({"name": file_name, "previous_name": existing_hashes[file_hash]})
            continue
        
        chunks = splitter.split_text(text)
        logger.info(f"[CUT] Split text {i+1} into {len(chunks)} chunks")
        
        for j, chunk in enumerate(chunks):
            all_chunks.append(Document(page_content=chunk, metadata={"file_hash": file_hash, "file_name": file_name}))
        
        successfully_ingested.append(file_name)

    if all_chunks:
        logger.info(f"[BRAIN] Embedding {len(all_chunks)} chunks into vector store")
        # Embedding progress
        batch_size = 5
        initial_count = vectorstore._collection.count()
        added_ids = []
        progress_bar = st.progress(0, text="Embedding documents...")
        for i in range(0, len(all_chunks), batch_size):
            batch = all_chunks[i : i + batch_size]
            ids = [f"{doc.metadata['file_hash']}_{idx}" for idx, doc in enumerate(batch, start=i)]
            vectorstore.add_documents(batch, ids=ids)
            added_ids.extend(ids)
            logger.info(f"[OK] Embedded batch {i//batch_size + 1}/{(len(all_chunks)-1)//batch_size + 1}")
            
            percent = int(((i + len(batch)) / len(all_chunks)) * 100)
            progress_bar.progress(min(percent, 99), text=f"Embedding documents...")
        
        time.sleep(0.5)
        progress_bar.empty()
        total_docs = vectorstore._collection.count()
        newly_added = total_docs - initial_count
        st.sidebar.info(f"[CHART] Total Saved Embeddings: {total_docs}")
        logger.info(f"[OK] Document ingestion complete. Total embeddings: {total_docs} (added {newly_added})")
        return newly_added, duplicate_files, successfully_ingested
    
    logger.warning("[WARN] No new chunks to embed")
    return 0, duplicate_files, successfully_ingested

def get_stored_files():
    """Retrieve list of all stored file names from Chroma DB."""
    try:
        logger.info("[RETRIEVE] Fetching stored files from vector store")
        vectorstore = get_vectorstore()
        docs_data = vectorstore.get()
        
        if not docs_data or not docs_data.get('metadatas'):
            logger.info("[EMPTY] No documents in vector store")
            return []
        
        # Get unique file names from metadata
        file_names = {}
        for meta in docs_data.get('metadatas', []):
            if meta is None:
                continue
            file_name = meta.get('file_name', 'Unknown')
            file_hash = meta.get('file_hash', '')
            if file_name and file_hash not in file_names:
                file_names[file_hash] = file_name
        
        unique_files = list(file_names.values())
        logger.info(f"[OK] Retrieved {len(unique_files)} unique file(s): {unique_files}")
        return sorted(unique_files)
    except Exception as e:
        logger.error(f"[ERROR] Error retrieving stored files: {str(e)}")
        return []

def delete_all_embeddings():
    """Delete all embeddings from the Chroma DB."""
    try:
        logger.info("[DELETE] Starting to delete all embeddings")
        vectorstore = get_vectorstore()
        
        # Count before deletion
        docs_before = vectorstore.get()
        count_before = len(docs_before['ids']) if docs_before and docs_before.get('ids') else 0
        logger.info(f"[COUNT] Embeddings before deletion: {count_before}")
        
        if count_before == 0:
            logger.warning("[WARN] No embeddings found to delete")
            return True
        
        # Get all document IDs
        all_ids = docs_before['ids']
        
        # DELETE all documents
        vectorstore.delete(ids=all_ids)
        logger.info(f"[OK] Successfully deleted {count_before} embeddings from vector store")
        
        # VERIFY deletion by checking count
        st.cache_resource.clear()
        vectorstore_new = get_vectorstore()
        docs_after = vectorstore_new.get()
        count_after = len(docs_after['ids']) if docs_after and docs_after.get('ids') else 0
        
        logger.info(f"[VERIFY] Embeddings count - Before: {count_before}, After: {count_after}")
        
        if count_after == 0:
            logger.info(f"[CONFIRMED] All embeddings successfully removed from chroma.sqlite3")
            return True
        else:
            logger.error(f"[FAILED] Deletion may not have fully persisted! Remaining: {count_after}")
            return False
    except Exception as e:
        logger.error(f"[ERROR] Error deleting embeddings: {str(e)}")
        return False

def delete_embedding_by_file_name(file_name: str):
    """Delete embeddings of a specific file by name."""
    try:
        logger.info(f"[DELETE] Deleting embeddings for file: {file_name}")
        vectorstore = get_vectorstore()
        
        # Count before deletion
        docs_before = vectorstore.get()
        count_before = len(docs_before['ids']) if docs_before and docs_before.get('ids') else 0
        
        # Get all documents
        if not docs_before or not docs_before.get('metadatas'):
            logger.warning("[WARN] No documents found")
            return False
        
        # Find IDs matching this file name
        ids_to_delete = []
        for idx, meta in enumerate(docs_before.get('metadatas', [])):
            if meta and meta.get('file_name') == file_name:
                ids_to_delete.append(docs_before['ids'][idx])
        
        if ids_to_delete:
            # DELETE from vectorstore
            vectorstore.delete(ids=ids_to_delete)
            logger.info(f"[OK] Successfully deleted {len(ids_to_delete)} embeddings for '{file_name}'")
            
            # VERIFY deletion by checking count
            st.cache_resource.clear()
            vectorstore_new = get_vectorstore()
            docs_after = vectorstore_new.get()
            count_after = len(docs_after['ids']) if docs_after and docs_after.get('ids') else 0
            
            logger.info(f"[VERIFY] Embeddings count - Before: {count_before}, After: {count_after}, Deleted: {count_before - count_after}")
            
            if count_after < count_before:
                logger.info(f"[CONFIRMED] Embeddings for '{file_name}' successfully removed from chroma.sqlite3")
                return True
            else:
                logger.error(f"[FAILED] Deletion may not have persisted to database!")
                return False
        else:
            logger.warning(f"[WARN] No embeddings found for '{file_name}'")
            return False
    except Exception as e:
        logger.error(f"[ERROR] Error deleting embeddings for {file_name}: {str(e)}")
        return False

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
        content_preview = doc.page_content[:50].replace('\n', ' ')
        file_name = doc.metadata.get('file_name', 'Unknown')
        logger.info(f"   {idx}. From '{file_name}': {content_preview}...")
    
    context = "\n\n".join([d.page_content for d in docs])
    prompt = f"""Answer the question using ONLY the provided context. 
If the answer cannot be found in the context, respond with 'NOT_FOUND'.

Context:
{context}

Question: {query}

Answer:"""
    logger.info("[BRAIN] Generating answer from LLM")
    answer = llm.invoke(prompt).strip()
    
    # Check if answer is valid
    if not answer or answer.upper() == "NOT_FOUND" or answer.lower().startswith("not found") or answer.lower().startswith("i cannot") or "cannot find" in answer.lower():
        logger.warning(f"[NOT_FOUND] LLM response indicates no answer in context: {answer[:50]}")
        return "NOT_FOUND"
    
    logger.info(f"[OK] Answer generated: {answer[:100]}...")
    return answer