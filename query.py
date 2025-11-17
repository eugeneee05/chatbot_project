import requests
import chromadb

DB_FOLDER = r"C:\Users\sw14\Documents\Internship\ChatBot_Project\chatbot_api\db"

EMBED_MODEL  = "text-embedding-nomic-embed-text-v1.5@q8_0"
ANSWER_MODEL = "phi-4-mini-instruct"

def embed(text):
    r = requests.post(
        "http://localhost:1234/v1/embeddings",
        json={"input": text, "model": EMBED_MODEL}
    )
    return r.json()["data"][0]["embedding"]

client = chromadb.PersistentClient(path=DB_FOLDER)
col = client.get_collection("pdf_knowledge")


THRESHOLD = 0.7  

while True:
    q = input("\nQ: ")
    if not q:
        break

    q_vec = embed(q)
    results = col.query(query_embeddings=[q_vec], n_results=8)

    distances = results["distances"][0]
    best_distance = distances[0]
    print(f"Best distance: {best_distance:.4f}")

    if best_distance > THRESHOLD:
        print("NOT FOUND")
        continue  


    context = "\n\n".join(results["documents"][0])

    prompt = f"""Use the context provided in the knowledge base only. Do not provide any extra explanation. Do not show another question, only reply to the question provided.

CONTEXT:
{context}

QUESTION:
{q}

ANSWER:
"""

    r = requests.post(
        "http://localhost:1234/v1/chat/completions",
        json={
            "model": ANSWER_MODEL,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.1
        }
    )

    print("→", r.json()["choices"][0]["message"]["content"])
