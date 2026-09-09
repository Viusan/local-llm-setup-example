import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

from langchain_core.messages import SystemMessage, HumanMessage

load_dotenv()

api_key = os.getenv("OSBOT_API_KEY")
base_url = os.getenv("OSBOT_BASE_URL")
model_name = os.getenv("LOCAL_MODEL")

llm = ChatOpenAI(
	api_key="dummy",
    	base_url=base_url,
    	default_headers={"x-api-key": api_key},
	model = model_name,
	temperature=0.3,
   	max_tokens=5000,
)

#first agent the writer
response = llm.invoke([
	SystemMessage("You are writer agent. Write a short paragraph (3-5 sentences) about the given topic in a humorus tone."),
	HumanMessage("Why is there snow in Antartica?"),
])

writer_response = response.content

#second agent the critic
critic_response = llm.invoke([
	SystemMessage("You are a serious teacher who hates humour. Review this paragraph and give 2-3 concrete pieces of feedback"),
	HumanMessage(writer_response)
])

#third agent the reviser
reviser_input = f"Original paragraph:\n{writer_response}\n\nFeedback:\n{critic_response.content}"
reviser_response = llm.invoke([
	SystemMessage("You are a reviser, given content and critic feedback rewrite the paragraph and only return the improved paragraph"),
	HumanMessage(reviser_input)
])

print(reviser_response.content)
