from flask import Flask, render_template, request
import os
from werkzeug.utils import secure_filename
from dotenv import load_dotenv

# =========================
# LOAD ENV
# =========================
load_dotenv()

# =========================
# IMPORTS
# =========================
from utils.pdf_loader import load_pdf
from utils.text_splitter import split_documents
from utils.embeddings import get_embeddings_model

from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain

from langchain_ollama import ChatOllama

# =========================
# GEMINI API KEY
# =========================

# =========================
# FLASK APP
# =========================
app = Flask(__name__)

UPLOAD_FOLDER = "uploads"

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# =========================
# GLOBAL VARIABLES
# =========================
vectorstore = None
qa_chain = None
summary = ""

# =========================
# HOME PAGE
# =========================
@app.route("/")
def home():
    return render_template("index.html")


# =========================
# UPLOAD PDF
# =========================
@app.route("/upload", methods=["POST"])
def upload_pdf():

    global vectorstore
    global qa_chain
    global summary

    try:

        print("STEP 1")

        if "pdf_file" not in request.files:
            return render_template(
                "index.html",
                error="No file selected"
            )

        file = request.files["pdf_file"]

        if file.filename == "":
            return render_template(
                "index.html",
                error="Please choose a PDF"
            )

        print("STEP 2")

        # =========================
        # SAVE PDF
        # =========================
        filename = secure_filename(file.filename)

        pdf_path = os.path.join(
            app.config["UPLOAD_FOLDER"],
            filename
        )

        file.save(pdf_path)

        print("PDF SAVED")

        # =========================
        # LOAD PDF
        # =========================
        documents = load_pdf(pdf_path)

        print("PDF LOADED")

        # =========================
        # SPLIT TEXT
        # =========================
        docs = split_documents(documents)

        print("TEXT SPLIT")

        # =========================
        # FULL TEXT
        # =========================
        full_text = ""

        for doc in docs:
            full_text += doc.page_content + "\n"

        # =========================
        # SUMMARY
        # =========================
        summary = full_text[:3000]

        print("SUMMARY CREATED")

        # =========================
        # EMBEDDINGS
        # =========================
        embeddings = get_embeddings_model()

        print("EMBEDDINGS READY")

        # =========================
        # VECTOR STORE
        # =========================
        vectorstore = FAISS.from_documents(
            docs,
            embeddings
        )

        print("VECTOR STORE READY")

        # =========================
        # NEW GEMINI MODEL
        # =========================
        llm = ChatOllama(
        model="phi3",
        temperature=0.3
        )

        # =========================
        # QA CHAIN
        # =========================
        qa_chain = load_qa_chain(
            llm,
            chain_type="stuff"
        )

        print("QA CHAIN READY")

        return render_template(
            "result.html",
            summary=summary,
            answer=""
        )

    except Exception as e:

        print("ERROR:", str(e))

        return render_template(
            "index.html",
            error=str(e)
        )


# =========================
# ASK QUESTION
# =========================
@app.route("/ask", methods=["POST"])
def ask_question():

    global vectorstore
    global qa_chain
    global summary

    try:

        question = request.form["question"]

        if vectorstore is None:
            return render_template(
                "result.html",
                summary=summary,
                answer="Please upload PDF first."
            )

        docs = vectorstore.similarity_search(
            question,
            k=3
        )

        response = qa_chain.run(
            input_documents=docs,
            question=question
        )

        return render_template(
            "result.html",
            summary=summary,
            answer=response
        )

    except Exception as e:

        print("QUESTION ERROR:", str(e))

        return render_template(
            "result.html",
            summary=summary,
            answer=f"Error: {str(e)}"
        )


# =========================
# MAIN
# =========================
if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=5000,
        debug=False
    )
    