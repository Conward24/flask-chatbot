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

# Fetch environment variables
pinecone_api_key = os.environ.get("PINECONE_API_KEY")
pinecone_environment = os.environ.get("PINECONE_ENVIRONMENT")
openai_api_key = os.environ.get("OPENAI_API_KEY")

if not pinecone_api_key or not pinecone_environment or not openai_api_key:
    logging.error("Missing required environment variables. Please check PINECONE_API_KEY, PINECONE_ENVIRONMENT, and OPENAI_API_KEY.")
    raise ValueError("One or more environment variables are missing.")

# Initialize OpenAI client
try:
    client = OpenAI(api_key=openai_api_key)
    logging.info("OpenAI client initialized successfully.")
except Exception as e:
    logging.error(f"Failed to initialize OpenAI client: {str(e)}")
    raise

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Define the Pinecone index name and host
index_name = "maternal-knowledge"
host = "https://maternal-knowledge-peybevm.svc.aped-4627-b74a.pinecone.io"

# Initialize Pinecone client
try:
    pinecone_instance = Pinecone(api_key=pinecone_api_key)
    pinecone_index = Index(index_name, host=host)
    logging.info(f"Connected to Pinecone index '{index_name}' at host '{host}'.")
except Exception as e:
    logging.error(f"Failed to initialize Pinecone: {str(e)}")
    raise

# Mock data for maternal topics
mock_results = [
    {"metadata": {"title": "Mock Title 1", "content": "Mock Content 1"}},
    {"metadata": {"title": "Mock Title 2", "content": "Mock Content 2"}}
]

# Helper function to search maternal topics
def search_topics(query):
    try:
        embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
        query_vector = embeddings.embed_query(query)
        logging.info(f"Query vector generated successfully: {query_vector}")

        # Mocking Pinecone results for testing
        return [
            {"title": res["metadata"]["title"], "content": res["metadata"]["content"]}
            for res in mock_results
        ]
    except Exception as e:
        logging.error(f"Error in search_topics: {str(e)}", exc_info=True)
        return []

# Helper function for empathetic responses
def get_random_empathetic_response(tag):
    responses = {
        "neutral": "I'm here to help in any way I can.",
        "happy": "That's wonderful to hear! How can I support you further?",
        "sad": "I'm sorry to hear that. Let me know how I can help."
    }
    return responses.get(tag, "I'm here to help in any way I can.")

@app.route('/query', methods=['POST'])
def query():
    try:
        data = request.get_json()
        user_message = data.get("query", "What are the signs of pregnancy?")
        emotion = data.get("emotion", "neutral")

        logging.info(f"Received query: {user_message} | Emotion: {emotion}")

        relevant_resources = search_topics(user_message)
        context = "\n\n".join([
            f"Title: {res['title']}\nContent: {res['content']}" for res in relevant_resources
        ])
        empathetic_phrase = get_random_empathetic_response(emotion)

        prompt = f"""
        The user asked: "{user_message}"
        Relevant Context:
        {context}

        Empathetic Response: {empathetic_phrase}
        Respond in an empathetic tone, directly addressing the user in the second person. Avoid switching to first person unless explicitly required by the query.
        """

        response = client.chat.completions.create(
            messages=[
                {"role": "system", "content": "You are an empathetic women’s health assistant specializing in maternal health."},
                {"role": "user", "content": prompt}
            ],
            model="gpt-3.5-turbo"
        )

        assistant_message = response.choices[0].message.content
        logging.info(f"AI response: {assistant_message}")

        return jsonify({"response": assistant_message})

    except Exception as e:
        logging.error(f"Error in /query: {str(e)}", exc_info=True)
        return jsonify({"error": str(e)}), 500

@app.route('/test-pinecone', methods=['POST'])
def test_pinecone():
    try:
        # Parse the incoming request
        data = request.get_json()
        query = data.get("query", "What are the signs of pregnancy?")
        
        logging.info(f"Test query received: {query}")
        
        # Generate the embedding for the query
        embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
        query_vector = embeddings.embed_query(query)
        logging.info(f"Generated query vector: {query_vector}")
        
        # Query Pinecone for relevant results
        results = pinecone_index.query(
            vector=query_vector,
            top_k=5,  # Number of top results to retrieve
            include_metadata=True
        )
        logging.info(f"Query results: {results}")
        
        # Format the results to return
        formatted_results = [
            {"title": match["metadata"]["title"], "content": match["metadata"]["content"]}
            for match in results["matches"]
        ]
        
        return jsonify({"results": formatted_results})
    
    except Exception as e:
        logging.error(f"Error during Pinecone test: {str(e)}", exc_info=True)
        return jsonify({"error": str(e)}), 500


@app.route('/', methods=['GET'])
def home():
    return jsonify({"message": "Welcome to the Flask Chatbot API. Use the /query endpoint to interact with the chatbot."})

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
