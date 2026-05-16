from langchain.chains import RetrievalQA

from langchain_google_genai import ChatGoogleGenerativeAI

GOOGLE_API_KEY = "AIzaSyAJLoZrWrJVoh7uAggnUyrSosnfxWQPxws"


def create_qa_chain(vectorstore):

    llm = ChatGoogleGenerativeAI(
        model="models/gemini-pro",
        google_api_key=GOOGLE_API_KEY,
        temperature=0.3
    )

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=vectorstore.as_retriever()
    )

    return qa_chain