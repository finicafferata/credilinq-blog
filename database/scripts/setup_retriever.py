#setup_retriever.py
import os
from dotenv import load_dotenv
from supabase import create_client, Client
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
import requests
import psycopg2
import numpy as np
import io

load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
SUPABASE_STORAGE_BUCKET = os.getenv("SUPABASE_STORAGE_BUCKET", "documents")
SUPABASE_DB_URL = os.getenv("SUPABASE_DB_URL")  # e.g. postgres://user:pass@host:port/dbname

# Initialize Supabase client
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# Function to process and embed a document from Supabase Storage
# file_path: path in Supabase Storage
# document_id: int, FK to documents table

def process_and_embed_document(file_path: str, document_id: int):
    """
    Downloads a file from Supabase Storage, splits it into chunks, generates embeddings,
    and inserts the chunks and embeddings into the document_chunks table in Supabase (pgvector).
    """
    # Download file from Supabase Storage
    print(f"Downloading {file_path} from Supabase Storage...")
    res = supabase.storage.from_(SUPABASE_STORAGE_BUCKET).download(file_path)
    if not res:
        raise Exception(f"File {file_path} not found in Supabase Storage.")
    file_content = res
    if isinstance(file_content, bytes):
        text = file_content.decode("utf-8")
    else:
        text = file_content.read().decode("utf-8")

    # Split document into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_text(text)
    print(f"Split into {len(splits)} chunks.")

    # Generate embeddings for each chunk
    embeddings_model = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
    print("Generating embeddings for chunks...")
    embeddings = embeddings_model.embed_documents(splits)

    # Insert chunks and embeddings into Supabase (pgvector)
    print("Inserting chunks and embeddings into Supabase (pgvector)...")
    conn = psycopg2.connect(SUPABASE_DB_URL)
    cur = conn.cursor()
    for chunk, embedding in zip(splits, embeddings):
        # Convert embedding to pgvector format
        embedding_str = '[' + ','.join([str(x) for x in embedding]) + ']'
        cur.execute(
            """
            INSERT INTO document_chunks (document_id, content, embedding)
            VALUES (%s, %s, %s)
            """,
            (document_id, chunk, embedding_str)
        )
    conn.commit()
    cur.close()
    conn.close()
    print(f"Inserted {len(splits)} chunks for document_id={document_id}.")

# Example usage (for testing):
if __name__ == "__main__":
    # Example: process_and_embed_document('sample_content.txt', 1)
    print("This script is now a library. Use process_and_embed_document(file_path, document_id) from your API.") 