SHL RAG Assistant

An AI-powered web application utilizing Retrieval-Augmented Generation (RAG) for intelligent, context-aware querying of product assessments.

📌 Overview

This project is an AI-powered web application that utilizes a Retrieval-Augmented Generation (RAG) architecture. It searches through a local knowledge base (data.csv) using FAISS vector search and generates intelligent, context-aware responses using Hugging Face's Inference API. It is built with Flask and includes deployment configurations for Vercel.

✨ Key Features

Interactive Chat Interface: A clean, responsive web UI for chatting with the assistant.

Contextual Retrieval (RAG): Uses sentence-transformers to embed queries and FAISS to retrieve the most relevant product/assessment information.

Hugging Face Integration: Generates natural language responses using open-source models (default: Mistral-7B-Instruct-v0.1).

Source Tracking: Allows users to toggle and view the exact data sources the AI used to generate its answer.

Serverless Ready: Pre-configured with vercel.json for easy deployment on Vercel.

🛠️ Project Structure

File / Directory

Description

app.py

The main Flask application handling routing, session management, and API endpoints.

data.csv

The primary dataset acting as the knowledge base for the application.

embedder.py

Handles text embeddings using the Hugging Face all-MiniLM-L6-v2 feature extraction pipeline.

retriever.py

Manages the FAISS vector index for fast and efficient similarity search.

hf_inference.py

Interacts with the Hugging Face Inference API to generate LLM responses.

templates/index.html

The frontend user interface for the interactive chat application.

requirements.txt

Python dependencies required to run the project.

vercel.json

Configuration settings for Vercel serverless deployment.

🚀 Quick Start

Prerequisites

Python 3.8 or higher

A Hugging Face Access Token

Installation

# Clone the repository
git clone <your-repo-url>
cd SHL-main/SHL

# Install dependencies (virtual environment recommended)
pip install -r requirements.txt


Environment Variables

Set your Hugging Face API token in your terminal (or create a .env file if you modify the code to load it):

# On Windows (Command Prompt)
set HF_API_TOKEN=your_huggingface_token_here

# On Mac/Linux
export HF_API_TOKEN="your_huggingface_token_here"


Running the Application

To start the local development server, run:

python app.py


Open your web browser and navigate to http://localhost:5000 to interact with the assistant.

🔌 API Endpoints

Endpoint

Method

Description

/health

GET

Health check endpoint to verify the server is running.

/recommend

POST

Accepts a JSON payload {"query": "your search text"} and returns a list of up to 10 recommended assessments based on vector similarity.

🌍 Deployment

This project is configured for serverless deployment on Vercel.

Install the Vercel CLI.

Run vercel in the root directory.

Add your HF_API_TOKEN to the Environment Variables in your Vercel project dashboard.
