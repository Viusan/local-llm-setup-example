#i am creating a demo to showcase rag.
#in reality we would have PDF papers but here i am just going to use a list to simulate a documents

documents = [
	"The mitochondria is the powerhouse of the cell. It generates ATP through cellular respiration.",
    	"Python's Global Interpreter Lock (GIL) prevents true multi-threaded execution of Python bytecode.",
    	"Network isolation limits what an autonomous AI agent can reach, reducing damage if it misbehaves.",
    	"TCP guarantees ordered, reliable delivery of data, while UDP is faster but unreliable.",
    	"RAG combines a retrieval step with a generation step so the model can answer using real source text.",

]

def retrieve(query: str, top_k: int = 2) -> list[str]:
	"""Return top_k documents most relevant to the query,
	using simple keyword overlap (no embedings yet)."""
	query_words = set(query.lower().split())

	scored = []
	for doc in documents:
		doc_words = set(doc.lower().split())
		overlap = len(query_words & doc_words)  #words in common between our search and what is in docs and turns to count, bigger count means more in common
		scored.append((overlap, doc)) #for each item in doc we score how much in common query and docs ex: {3: "TCP guaren...}

	scored.sort(key=lambda pair: pair[0], reverse=True) #best match first, pair[0] looks at score, and reverse makes highest first

	result = []
	for score, doc in scored[:top_k]: #return top_k matches which is defaulted to 2 unless we change, and since :top_k its slicing so we say get result up to top_k (two results here if there are that many in the first place)
		if score > 0:
			result.append(doc)
	return result

if __name__ == "__main__":
    results = retrieve("give fun fact about Python")
    for r in results:
        print(r)
