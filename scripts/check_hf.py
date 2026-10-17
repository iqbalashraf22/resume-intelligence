from app.parsing.llm_parser import get_client

client = get_client()
response = client.chat_completion(
    messages=[{"role": "user", "content": "Reply with exactly: hello"}],
    max_tokens=20,
)
print(response.choices[0].message.content)