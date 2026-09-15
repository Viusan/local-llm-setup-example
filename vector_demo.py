#rag just checks for overlapping words which is not enough at all cases
#if we ask whats the weather like today, and in our pdf it says Oslo is sunny. There are no overlapping words
#vector database basically has many connections to similar meaning words, like weather and sunny

from sentence_transformers import SentenceTransformer
import chromadb

documents = [
        "The mitochondria is the powerhouse of the cell. It generates ATP through cellular respiration.",
        "Python's Global Interpreter Lock (GIL) prevents true multi-threaded execution of Python bytecode.",
        "Network isolation limits what an autonomous AI agent can reach, reducing damage if it misbehaves.",
        "TCP guarantees ordered, reliable delivery of data, while UDP is faster but unreliable.",
        "RAG combines a retrieval step with a generation step so the model can answer using real source text.",
        "Viusan studies computer engineering at OsloMet",
        "Sanchay studies computer engineering at OsloMet",
        "Elias has birthday in November and studies at OsloMet",
        "Viusan is 21 year old",
        "Oslo is currently sunny",
        "OsloMet is located at pilestredet",
        "Nationaltheateret is close to OsloMet",
]

#load the local embedding model (downloads the weights once, then it's cached)
embedder = SentenceTransformer("all-MiniLM-L6-v2")

#create chroma collection and load my documents as embeddings

#in-memory vector db, nothing gets saved to disk, so each time i run its new reset
client = chromadb.Client()
collection = client.create_collection(name="documents")

#turn all the documents into embedings at once
embeddings = embedder.encode(documents).tolist()

#store to collection
collection.add(
	ids=[str(i) for i in range(len(documents))], #chroma needs unique id per entry
	embeddings=embeddings,
	documents=documents,
)

#similar retrieve function that we also used in the rag demo
#quick thing to note: top_k = 5 here, which means we will get 5 results back. So we can get unreleated stuff aswell
#NOTE WE CAN ADD A THRESHOLD FILTER TO "FIX" this
def retrieve(query: str, top_k: int = 5) -> list[str]:
    """Return the top_k documents most relevant to the query,
    using vector similarity search instead of keyword overlap."""
    query_embedding = embedder.encode([query]).tolist()  # embed the query the same way

    results = collection.query(
        query_embeddings=query_embedding,
        n_results=top_k,
    )

    return results["documents"][0]  # results for our one query

if __name__ == "__main__":
    matches = retrieve("what do you know about studies and universities")
    for m in matches:
        print(m)
