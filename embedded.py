import os
import re
import pdfplumber
import requests
import chromadb
import numpy as np

PDF_FOLDER = r"C:\Users\sw14\Documents\Internship\ChatBot_Project\chatbot_api\knowledge_base"
DB_FOLDER  = r"C:\Users\sw14\Documents\Internship\ChatBot_Project\chatbot_api\db"

EMBED_MODEL = "text-embedding-nomic-embed-text-v1.5@q8_0"

CHUNK_SIZE = 256
CHUNK_OVERLAP = 50

client = chromadb.PersistentClient(path=DB_FOLDER)
col = client.get_or_create_collection("pdf_knowledge")

def clean_text(text):
    """Keep natural sentence structure, remove weird whitespace or non-ASCII characters."""
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^\x00-\x7F]+', ' ', text)  
    return text.strip()

def chunk(text, size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    """Split text into overlapping chunks."""
    chunks = []
    start = 0
    while start < len(text):
        end = start + size
        chunk = text[start:end]
        if len(chunk.split()) >= 30:
            chunks.append(chunk)
        start += size - overlap
    return chunks

def embed(text):
    """Get embedding from local model and normalize."""
    r = requests.post(
        "http://localhost:1234/v1/embeddings",
        json={"input": text, "model": EMBED_MODEL}
    )
    vec = r.json()["data"][0]["embedding"]
    vec = np.array(vec)
    vec /= np.linalg.norm(vec) + 1e-10
    return vec.tolist()

for fname in os.listdir(PDF_FOLDER):
    if not fname.lower().endswith(".pdf"):
        continue

    fullpath = os.path.join(PDF_FOLDER, fname)
    print("Processing:", fname)

    all_text = ""
    with pdfplumber.open(fullpath) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text() or ""
            all_text += page_text + "\n"

    cleaned_text = clean_text(all_text)
    pieces = chunk(cleaned_text)

    for i, c in enumerate(pieces):
        vector = embed(c)
        uid = f"{fname}-{i}"

        # Store better metadata
        metadata = {
            "source": fname,
            "chunk_index": i,
            "file_path": fullpath
        }

        col.add(
            ids=[uid],
            embeddings=[vector],
            documents=[c],
            metadatas=[metadata]
        )

print("Done.")
