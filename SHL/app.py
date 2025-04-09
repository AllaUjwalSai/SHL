from flask import Flask, render_template, request, session, redirect, url_for, jsonify
import pandas as pd
from embedder import embed_texts
from retriever import Retriever
from hf_inference import run_llm
import os
import re

# Initialize Flask app
app = Flask(__name__)
app.secret_key = os.urandom(24)  # Required for sessions

# Load CSV data
data_df = pd.read_csv("data.csv")

# Clean the Assessment Length field to extract numeric values
def clean_duration(duration_str):
    if pd.isna(duration_str):
        return 60  # Default value if missing
    
    # Extract first number found in the string
    match = re.search(r'\d+', str(duration_str))
    return int(match.group()) if match else 60

data_df['duration'] = data_df['Assessment Length'].apply(clean_duration)

def build_context(row):
    context = (
        f"Product Name: {row['Product Name']}\n"
        f"Description: {row['Description']}\n"
        f"Job Levels: {row['Job Levels']}\n"
        f"Languages: {row['Languages']}\n"
        f"Duration: {row['duration']} minutes\n"
        f"Test Types: {row['Test Types']}\n"
        f"Remote Testing: {row['Remote Testing']}\n"
    )
    return context

documents = data_df.apply(build_context, axis=1).tolist()
doc_embeddings = embed_texts(documents)
retriever = Retriever(dim=384)
retriever.build_index(doc_embeddings, documents)

# ----------------- API Endpoints -----------------
@app.route("/health", methods=["GET"])
def health_check():
    return jsonify({"status": "healthy"}), 200

@app.route("/recommend", methods=["POST"])
def recommend_assessments():
    data = request.get_json()
    query = data.get("query", "")

    if not query:
        return jsonify({"error": "No query provided"}), 400

    query_embedding = embed_texts([query])[0]
    top_docs = retriever.query(query_embedding, k=10)  # Get up to 10 recommendations

    recommended_assessments = []
    for doc in top_docs:
        try:
            index = documents.index(doc)
            row = data_df.iloc[index]
            
            # Construct assessment entry according to specification
            assessment = {
                "url": str(row['Product URL']),
                "adaptive_support": "No",  # Default value as this field isn't in CSV
                "description": str(row['Description'])[:500],  # Limit description length
                "duration": int(row['duration']),  # Ensure JSON-safe integer
                "remote_support": str(row['Remote Testing']),
                "test_type": [str(t).strip() for t in str(row['Test Types']).split(',')]
            }
            recommended_assessments.append(assessment)
        except Exception as e:
            app.logger.error(f"Error processing document: {e}")
            continue


    # Ensure minimum 1 and maximum 10 assessments
    if not recommended_assessments:
        recommended_assessments.append({
            "url": "https://www.shl.com/default-assessment",
            "adaptive_support": "No",
            "description": "No specific assessment found for your query. Please try different keywords.",
            "duration": 60,
            "remote_support": "Yes",
            "test_type": ["General"]
        })
    else:
        recommended_assessments = recommended_assessments[:10]

    return jsonify({"recommended_assessments": recommended_assessments}), 200

# ----------------- Web Interface Routes -----------------
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
    app.run(host='0.0.0.0', debug=True)
