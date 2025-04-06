from flask import Flask, render_template, request, session, redirect, url_for
import pandas as pd
from embedder import embed_texts
from retriever import Retriever
from hf_inference import run_llm
import os

# Initialize Flask app
app = Flask(__name__)
app.secret_key = os.urandom(24)  # Required for sessions

# Load CSV data
data_df = pd.read_csv("data.csv")

def build_context(row):
    context = (
        f"Product Name: {row['Product Name']}\n"
        f"Description: {row['Description']}\n"
        f"Job Levels: {row['Job Levels']}\n"
        f"Languages: {row['Languages']}\n"
        f"Assessment Length: {row['Assessment Length']}\n"
        f"Test Types: {row['Test Types']}\n"
        f"Remote Testing: {row['Remote Testing']}\n"
    )
    return context

documents = data_df.apply(build_context, axis=1).tolist()
doc_embeddings = embed_texts(documents)
retriever = Retriever(dim=384)
retriever.build_index(doc_embeddings, documents)

@app.route("/", methods=["GET", "POST"])
def index():
    if "chat_history" not in session:
        session["chat_history"] = []

    if request.method == "POST":
        query = request.form["query"]
        query_embedding = embed_texts([query])[0]
        top_docs = retriever.query(query_embedding, k=5)
        context = "\n\n".join(top_docs)
        prompt = f"Context:\n{context}\n\nQuestion: {query}\nAnswer:"
        generated_text = run_llm(prompt)
        if "Answer:" in generated_text:
            answer = generated_text.split("Answer:")[-1].strip()
        else:
            answer = generated_text.strip()

        session["chat_history"].append({
            "query": query,
            "answer": answer,
            "context": context
        })
        session.modified = True

    return render_template("index.html", chat_history=session["chat_history"])

@app.route("/reset", methods=["POST"])
def reset_chat():
    session.pop("chat_history", None)
    return redirect(url_for("index"))

if __name__ == '__main__':
    app.run(debug=True)
