import json
import logging
import random
import os
from flask import Flask, request, jsonify
from flask_cors import CORS
import pinecone
from langchain_community.embeddings import OpenAIEmbeddings
from openai import OpenAI

# Configure logging
logging.basicConfig(
    filename="app.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# Initialize OpenAI client
client = OpenAI(
    api_key=os.environ.get("OPENAI_API_KEY")  # Environment variable for API key
)

# Initialize the Flask app
app = Flask(__name__)
CORS(app)

# Load resources and empathetic responses
try:
    with open("structured_maternal_guide.json", "r", encoding="utf-8") as f:
        maternal_data = json.load(f)

    with open("empathetic_responses.json", "r", encoding="utf-8") as f:
        empathetic_responses = json.load(f)
except FileNotFoundError as e:
    logging.error(f"File not found: {e}")
    maternal_data = []
    empathetic_responses = []

# Pinecone Initialization
index_name = "maternal-knowledge"
pinecone.init(
    api_key=os.environ.get("PINECONE_API_KEY"),
    environment=os.environ.get("PINECONE_ENVIRONMENT")
)

# Check if the index exists; create it if it doesn't
if index_name not in pinecone.list_indexes():
    pinecone.create_index(
        name=index_name,
        dimension=1536,
        metric='cosine'
    )

# Connect to the index
pinecone_index = pinecone.Index(index_name)

# Helper function to search maternal topics
def search_topics(query):
    embeddings = OpenAIEmbeddings(openai_api_key=os.environ.get("OPENAI_API_KEY"))
    query_vector = embeddings.embed_query(query)
    results = pinecone_index.query(query_vector, top_k=5, include_metadata=True)
    return [{"title": res["metadata"]["title"], "content": res["metadata"]["content"]} for res in results["matches"]]

# Other endpoints and logic remain unchanged...
