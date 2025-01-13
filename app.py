import json
import logging
import random
import os
from flask import Flask, request, jsonify
from flask_cors import CORS
from pinecone import Pinecone, ServerlessSpec, Index
from langchain_openai import OpenAIEmbeddings
from openai import OpenAI

# Configure logging
logging.basicConfig(
    filename="app.log",
    level=logging.DEBUG,  # DEBUG level for detailed logs
    format="%(asctime)s - %(levelname)s - %(message)s",
    force=True  # Ensure global logging config
)

# Validate and log environment variables
pinecone_api_key = os.environ.get("PINECONE_API_KEY")
pinecone_environment = os.environ.get("PINECONE_ENVIRONMENT")
openai_api_key = os.environ.get("OPENAI_API_KEY")

logging.info(f"PINECONE_API_KEY (DEBUG): {pinecone_api_key}")  # Debug log for API key (ensure it's removed later)
logging.info(f"PINECONE_ENVIRONMENT: {pinecone_environment}")
logging.info(f"OPENAI_API_KEY: {'SET' if openai_api_key else 'NOT SET'}")

if not pinecone_api_key or not pinecone_environment:
    raise ValueError("PINECONE_API_KEY or PINECONE_ENVIRONMENT is not set in environment variables.")

# Initialize OpenAI client
client = OpenAI(api_key=openai_api_key)

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

# Define the Pinecone index name
index_name = "maternal-knowledge"
host = "https://maternal-knowledge-peybevm.svc.aped-4627-b74a.pinecone.io"

# Initialize Pinecone client with error handling
try:
    logging.info("Initializing Pinecone client...")
    pinecone_instance = Pinecone(api_key=pinecone_api_key)
    logging.info("Pinecone client initialized successfully.")

    # Check if the index exists; create it if it doesn't
    if index_name not in pinecone_instance.list_indexes().names():
        logging.info(f"Creating index '{index_name}'...")
        pinecone_instance.create_index(
            name=index_name,
            dimension=1536,
            metric='cosine',
            spec=ServerlessSpec(
                cloud='aws',
                region=pinecone_environment
            )
        )
        logging.info(f"Index '{index_name}' created successfully.")

    # Connect to the index with explicit host
    pinecone_index = Index(index_name, host=host)
    logging.info(f"Successfully connected to Pinecone index '{index_name}' at host: {host}")

except Exception as e:
    logging.error(f"Failed to initialize Pinecone: {str(e)}", exc_info=True)
    raise

# Helper function to search maternal topics
def search_topics(query):
    try:
        embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
        query_vector = embeddings.embed_query(query)
        logging.info(f"Query vector generated successfully: {query_vector}")

        # Use keyword arguments for Pinecone query
        results = pinecone_index.query(
            vector=query_vector,
            top_k=5,
            include_metadata=True
        )
        logging.info(f"Query results from Pinecone: {results}")

        return [{"title": res["metadata"]["title"], "content": res["metadata"]["content"]} for res in results["matches"]]
    except Exception as e:
        logging.error(f"Error in search_topics: {str(e)}", exc_info=True)
        raise

# Helper function to retrieve a random empathetic response based on tags (emotions)
def get_random_empathetic_response(tag):
    logging.info(f"Emotion tag received: {tag}")
    if tag is None:
        tag = "neutral"
        logging.warning("Received a None tag; defaulting to 'neutral'.")

    matches = [
        entry for entry in empathetic_responses
        if entry.get("tags", "").lower() == tag.lower()
    ]
    if matches:
        return random.choice(matches)["utterance"]  # Use random.choice to pick a response

    logging.warning(f"No matches found for tag: {tag}; returning default response.")
    return "I'm here to help in any way I can."

@app.route('/query', methods=['POST'])
def query():
    try:
        # Parse the incoming request
        data = request.get_json()
        user_message = data.get("query", "What are the signs of pregnancy?")
        topic = data.get("topic", None)
        emotion = data.get("emotion", "neutral")

        logging.info(f"Received query: {user_message} | Topic: {topic} | Emotion: {emotion}")

        # Retrieve relevant maternal resources and empathetic response
        relevant_resources = search_topics(user_message)
        context = "\n\n".join([
            f"Title: {res['title']}\nContent: {res['content']}" for res in relevant_resources
        ])
        empathetic_phrase = get_random_empathetic_response(emotion)

        # Construct the OpenAI prompt
        prompt = f"""
        The user asked: "{user_message}"
        Relevant Context:
        {context}

        Empathetic Response: {empathetic_phrase}
        Respond in an empathetic tone, directly addressing the user in the second person. Avoid switching to first person unless explicitly required by the query.
        """

        # Call OpenAI API
        response = client.chat_completions.create(
            messages=[
                {"role": "system", "content": "You are an empathetic and culturally inclusive women’s health assistant who specializes in maternal health and directly answers the user's questions in the second person, focusing on their needs and concerns."},
                {"role": "user", "content": prompt}
            ],
            model="gpt-3.5-turbo"  # Replace with "gpt-4" if applicable
        )

        assistant_message = response.choices[0].message.content
        logging.info(f"AI response: {assistant_message}")

        return jsonify({"response": assistant_message})

    except Exception as e:
        logging.error(f"Error occurred in /query: {str(e)}", exc_info=True)
        return jsonify({"error": str(e)}), 500

@app.route('/test-pinecone', methods=['POST'])
def test_pinecone():
    try:
        data = request.get_json()
        query = data.get("query", "What are the signs of pregnancy?")
        
        # Debug: Log the query being tested
        logging.info(f"Test query received: {query}")
        
        # Perform a Pinecone query
        embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
        query_vector = embeddings.embed_query(query)
        logging.info(f"Generated query vector: {query_vector}")
        
        results = pinecone_index.query(
            vector=query_vector,
            top_k=5,
            include_metadata=True
        )
        logging.info(f"Pinecone query results: {results}")
        
        return jsonify({
            "results": [
                {"title": res["metadata"]["title"], "content": res["metadata"]["content"]}
                for res in results["matches"]
            ]
        })

    except Exception as e:
        # Log the full stack trace for debugging
        logging.error(f"Error during Pinecone test: {str(e)}", exc_info=True)
        return jsonify({"error": str(e)}), 500

@app.route('/', methods=['GET'])
def home():
    return jsonify({"message": "Welcome to the Flask Chatbot API. Use the /query endpoint to interact with the chatbot."})

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000)
