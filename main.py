import os
from dotenv import load_dotenv
from openai import OpenAI
from langsmith.wrappers import wrap_openai

load_dotenv()

api_key = os.getenv("OSBOT_API_KEY")
base_url = os.getenv("OSBOT_BASE_URL")
model_name = os.getenv("LOCAL_MODEL")

print(f"API key found: {'yes' if api_key else 'NO'}")
print(f"Base URL: {base_url}")
print(f"Model: {model_name}")

client = wrap_openai(OpenAI(
    api_key="dummy",
    base_url=base_url,
    default_headers={"x-api-key": api_key},
))

response = client.chat.completions.create(
    model=model_name,
    messages=[{"role": "user", "content": "Reply with exactly: API key works."}],
    max_tokens=300,
)
print(response.choices[0].message)
print(response.usage)
