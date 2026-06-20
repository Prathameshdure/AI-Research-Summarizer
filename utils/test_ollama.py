from langchain_ollama import ChatOllama

llm = ChatOllama(
    model="phi3",
    temperature=0.3
)

response = llm.invoke("Hello! Introduce yourself in one sentence.")

print(response.content)