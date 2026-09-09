import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.tools import tool #needed to create tools
from langchain_core.messages import ToolMessage #needed to generate a response message from tool usage

load_dotenv()

api_key = os.getenv("OSBOT_API_KEY")
base_url = os.getenv("OSBOT_BASE_URL")
model_name = os.getenv("LOCAL_MODEL")

llm = ChatOpenAI(
	api_key="dummy",
	base_url=base_url,
	default_headers={"x-api-key": api_key},
	model=model_name,
	temperature=0.3,
	max_tokens=3000,
)

@tool
def calculator(a: float, b: float, operation: str) -> str:
	"""Perform a basic arithmetic operation (+, -, /) on two numbers."""
	if operation == "-":
		return str(a-b)
	elif operation == "+":
		return str(a+b)
	elif operation == "/":
		return str(a/b)
	else:
		return "unknown operation"

#bind the tools to llm, this just lets the llm know that this tool exists and what is does
llm_with_tools = llm.bind_tools([calculator])

messages = [
	SystemMessage("You are a helpful assistant with access to a calculator tool"),
	HumanMessage("What is 9+10"),
]

response = llm_with_tools.invoke(messages)
messages.append(response) #keep the tool call AI message in history

for tool_call in response.tool_calls:
	if tool_call["name"] == "calculator":
		args = tool_call["args"] #this is dict of parameter values the model decided to pass into the tool
		#here is where we actually run our calculator function
		result = calculator.invoke(args) #look like this for example: calculator(a=9, b=10, operation="+")

		#now we have to feed the result back with the matching tool id so it knows which call this response answers
		messages.append(
			ToolMessage(content=str(result), tool_call_id=tool_call["id"])
		)

#now we ask the model again to run but this time we have the tool result as context aswell
final_response = llm_with_tools.invoke(messages)
print(final_response.content)
