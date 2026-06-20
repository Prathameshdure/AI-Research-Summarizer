from langchain.chains import RetrievalQA
from langchain_ollama import ChatOllama


def create_qa_chain(vectorstore):

    llm = ChatOllama(
        model="phi3",
        temperature=0.3,
    )

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        chain_type="stuff"
    )

    return qa_chain