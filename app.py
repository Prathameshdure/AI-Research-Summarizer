from flask import Flask, render_template, request
import os
from werkzeug.utils import secure_filename

# PDF Loader
from langchain_community.document_loaders import PyPDFLoader

# Text Splitter
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Embeddings
from langchain_community.embeddings import HuggingFaceEmbeddings

# Vector Store
from langchain_community.vectorstores import FAISS

# Ollama LLM
from langchain_community.llms import Ollama

# QA Chain
from langchain.chains.question_answering import load_qa_chain

app = Flask(__name__)

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Global variables
vectorstore = None
qa_chain = None
summary = ""


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/upload", methods=["POST"])
def upload():

    global vectorstore
    global qa_chain
    global summary

    try:
        print("STEP 1")

        if "pdf" not in request.files:
            return render_template(
                "result.html",
                summary="No PDF uploaded.",
                answer=""
            )

        file = request.files["pdf"]

        if file.filename == "":
            return render_template(
                "result.html",
                summary="Please select a PDF file.",
                answer=""
            )

        print("STEP 2")

        filename = secure_filename(file.filename)
        filepath = os.path.join(UPLOAD_FOLDER, filename)

        file.save(filepath)

        print("PDF SAVED")

        # Load PDF
        loader = PyPDFLoader(filepath)
        documents = loader.load()

        print("PDF LOADED")

        # Split Text
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )

        docs = text_splitter.split_documents(documents)

        print("TEXT SPLIT")

        # Create Summary
        full_text = ""

        for doc in docs[:5]:
            full_text += doc.page_content

        summary = full_text[:2000]

        print("SUMMARY CREATED")

        # Embeddings
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )

        print("EMBEDDINGS READY")

        # Vector Store
        vectorstore = FAISS.from_documents(docs, embeddings)

        print("VECTOR STORE READY")

        # Ollama Model
        llm = Ollama(model="phi3")

        print("OLLAMA READY")

        # QA Chain
        qa_chain = load_qa_chain(llm, chain_type="stuff")

        print("QA CHAIN READY")

        return render_template(
            "result.html",
            summary=summary,
            answer=""
        )

    except Exception as e:
        print("ERROR:", str(e))

        return render_template(
            "result.html",
            summary=f"Error: {str(e)}",
            answer=""
        )


@app.route("/ask", methods=["POST"])
def ask():

    global vectorstore
    global qa_chain
    global summary

    try:
        question = request.form["question"]

        print("QUESTION:", question)

        if vectorstore is None:
            return render_template(
                "result.html",
                summary="Please upload a PDF first.",
                answer=""
            )

        # Similarity Search
        docs = vectorstore.similarity_search(question, k=3)

        print("DOCUMENTS FOUND")

        # Generate Answer
        response = qa_chain.run(
            input_documents=docs,
            question=question
        )

        print("ANSWER GENERATED")

        return render_template(
            "result.html",
            summary=summary,
            answer=response
        )

    except Exception as e:
        print("ASK ERROR:", str(e))

        return render_template(
            "result.html",
            summary=summary,
            answer=f"Error: {str(e)}"
        )


if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=5000,
        debug=False,
        use_reloader=False
    )