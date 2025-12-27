# ingest.py

import os
import argparse
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
# The HuggingFaceBgeEmbeddings class is a wrapper, we'll use a more generic one for broader model support
from langchain_community.embeddings import HuggingFaceEmbeddings

print("--- Script starting ---")

# --- Configuration ---
DB_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "db", "chroma_db")
# --- NEW MODEL ---
# This model is small, fast, and has good multilingual capabilities.
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2" 

def main(doc_path, collection_name):
    print(f"[main function] Starting ingestion process for document: {doc_path}")
    print(f"[main function] Target ChromaDB collection: {collection_name}")

    # 1. Load the document
    try:
        loader = TextLoader(doc_path, encoding='utf-8')
        documents = loader.load()
        print(f"[main function] Successfully loaded {len(documents)} document(s).")
    except Exception as e:
        print(f"[main function] Error loading document: {e}")
        return

    # 2. Split the document into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800, # Smaller chunk size can be beneficial for smaller models
        chunk_overlap=150
    )
    chunks = text_splitter.split_documents(documents)
    print(f"[main function] Split document into {len(chunks)} chunks.")

    # 3. Initialize the embedding model
    print(f"[main function] Loading embedding model: {EMBEDDING_MODEL_NAME}")
    # Use the more generic HuggingFaceEmbeddings class
    model_kwargs = {'device': 'cpu'} 
    encode_kwargs = {'normalize_embeddings': True} # This is important for MiniLM
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs
    )
    print("[main function] Embedding model loaded.")

    # 4. Create and persist the vector store
    print(f"[main function] Creating vector store in: {DB_DIR}")
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name=collection_name,
        persist_directory=DB_DIR
    )
    print("-" * 50)
    print("✅ Ingestion complete!")
    print(f"Vector store for client '{collection_name}' is ready and saved to disk.")
    print("-" * 50)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Ingest a document into a ChromaDB collection."
    )
    parser.add_argument(
        "doc_path", 
        type=str, 
        help="Path to the document file (e.g., 'data/qair_aviation_group.txt')."
    )
    parser.add_argument(
        "collection_name", 
        type=str, 
        help="Unique name for the client's collection (e.g., 'qair_aviation')."
    )
    args = parser.parse_args()
    
    if not os.path.exists(args.doc_path):
        print(f"🛑 CRITICAL ERROR: Document path '{args.doc_path}' does not exist!")
    else:
        main(args.doc_path, args.collection_name)
