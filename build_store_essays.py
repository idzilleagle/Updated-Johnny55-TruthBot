# --- build_store_essays.py (Optimized for Local Speed) ---

import os
from dotenv import load_dotenv
# Note: time module is no longer needed

# --- LangChain Imports ---
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

# Import your essay database
# Ensure essay_data.py is in the same folder
from essay_data import ESSAY_DATABASE

# Load environment variables
load_dotenv()

# --- Configuration ---
FAISS_INDEX_PATH = "faiss_index"

def create_vector_store():
    """
    The main function to build and save the vector store from essay_data.py
    """
    
    # 1. Load all essays from essay_data.py
    print("--- Starting Knowledge Base Build ---")
    print("Step 1: Loading essays from essay_data.py...")
    all_docs = []
    
    for essay_id, essay_info in ESSAY_DATABASE.items():
        title = essay_info.get("title", "Untitled")
        author = essay_info.get("author", "Unknown")
        content = essay_info.get("content", "")
        tags = essay_info.get("tags", [])
        
        if content.strip():
            metadata = {
                "source": f"Essay {essay_id}: {title}",
                "essay_id": essay_id,
                "title": title,
                "author": author,
                "tags": ", ".join(tags) if tags else ""
            }
            
            doc = Document(page_content=content, metadata=metadata)
            all_docs.append(doc)
    
    if not all_docs:
        print("Error: No essays found in ESSAY_DATABASE. Halting process.")
        return
        
    print(f"Successfully loaded {len(all_docs)} essays.")

    # 2. Split the loaded documents into smaller chunks
    print("\nStep 2: Splitting essays into manageable chunks...")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=200)
    chunked_docs = text_splitter.split_documents(all_docs)
    print(f"Split essays into {len(chunked_docs)} chunks.")

    # 3. Initialize the embeddings model
    print("\nStep 3: Initializing Local Embeddings model...")
    # CRITICAL: This must match the model used in bot.py
    embeddings = HuggingFaceEmbeddings(
        model_name="paraphrase-MiniLM-L3-v2",
        model_kwargs={'device': 'cpu'}
    )

    # 4. Build the FAISS index
    print("\nStep 4: Creating FAISS index...")
    print("Processing locally (no API limits). This may take a moment depending on your CPU...")
    
    # Since we are running locally, we don't need to batch or sleep!
    # We can process everything at once.
    vector_store = FAISS.from_documents(documents=chunked_docs, embedding=embeddings)

    # 5. Save the completed index to disk
    print("\nStep 5: Saving final index to disk...")
    vector_store.save_local(FAISS_INDEX_PATH)
    
    print("\n--- Build Complete! ---")
    print(f"Vector store has been saved to the '{FAISS_INDEX_PATH}' folder.")
    print(f"Total essays indexed: {len(all_docs)}")
    print(f"Total chunks created: {len(chunked_docs)}")

if __name__ == "__main__":
    create_vector_store()