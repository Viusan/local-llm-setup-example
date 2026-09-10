#i am creating a demo to showcase rag.
#in reality we would have PDF papers but here i am just going to use a list to simulate a documents

import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_core.messages import SystemMessage, HumanMessage, ToolMessage

load_dotenv()

api_key = os.getenv("OSBOT_API_KEY")
base_url = os.getenv("OSBOT_BASE_URL")
model_name = os.getenv("LOCAL_MODEL")

documents = [
	"The mitochondria is the powerhouse of the cell. It generates ATP through cellular respiration.",
    	"Python's Global Interpreter Lock (GIL) prevents true multi-threaded execution of Python bytecode.",
    	"Network isolation limits what an autonomous AI agent can reach, reducing damage if it misbehaves.",
    	"TCP guarantees ordered, reliable delivery of data, while UDP is faster but unreliable.",
    	"RAG combines a retrieval step with a generation step so the model can answer using real source text.",

]

llm = ChatOpenAI(
    base_url=base_url,
    api_key="dummy",  # osbot ignores this field
    default_headers={"x-api-key": api_key},  # this is what osbot actually checks
    model=model_name,
    max_tokens=500,
    temperature=0.3,
)

messages = [
	SystemMessage("You are a helpful assistant with access to a document search tool."),
	HumanMessage("What is a fun fact about python?"),
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

@tool
def search_document(query: str) -> str:
	"""Search the document collection for information relevant to the query."""
	results = retrieve(query)
	if not results:
		return "No relevant documents found"
	return "\n\n".join(results) #returns the stuff found in strings each on new row

llm_with_tools = llm.bind_tools([search_document])

response = llm_with_tools.invoke(messages)
messages.append(response) #keep ai took call history

for tool_call in response.tool_calls:
	if tool_call["name"] == "search_document":
		args = tool_call["args"]
		result = search_document.invoke(args) #run the search
		messages.append( #append the result/content to our messages
            		ToolMessage(content=result, tool_call_id=tool_call["id"])
        	)

final_response = llm_with_tools.invoke(messages) #now we run llm again with all the content to get a response
print(final_response.content)
