import json
import logging
import random  # Ensure random is imported
import os
from flask import Flask, request, jsonify
from openai import OpenAI  # Import OpenAI client

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

# Load resources and empathetic responses
try:
    with open("resources.json", "r") as f:
        resources = json.load(f)

    with open("empathetic_responses.json", "r") as f:
        empathetic_responses = json.load(f)
except FileNotFoundError as e:
    logging.error(f"File not found: {e}")
    resources = []
    empathetic_responses = []

# Helper function to retrieve resources for a specific topic
def get_relevant_resources(topic):
    return [res for res in resources if res["topic"].lower() == topic.lower()]

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
        user_message = data.get("query", "Write a haiku about AI.")
        topic = data.get("topic", "General")
        emotion = data.get("emotion", "neutral")

        logging.info(f"Received query: {user_message} | Topic: {topic} | Emotion: {emotion}")

        # Retrieve relevant resources and empathetic response
        relevant_resources = get_relevant_resources(topic)
        context = "\n".join([f"{res['content']} (Source: {res['source']})" for res in relevant_resources])
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
        response = client.chat.completions.create(
            messages=[
                {"role": "system", "content": "You are an empathetic and culturally inclusive assistant who directly answers the user's questions in the second person, focusing on their needs and concerns."},
                {"role": "user", "content": prompt}
            ],
            model="gpt-3.5-turbo"  # Replace with "gpt-4" if applicable
        )

        assistant_message = response.choices[0].message.content
        logging.info(f"AI response: {assistant_message}")

        return jsonify({"response": assistant_message})

    except Exception as e:
        logging.error(f"Error occurred: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/', methods=['GET'])
def home():
    return jsonify({"message": "Welcome to the Flask Chatbot API. Use the /query endpoint to interact with the chatbot."})

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000)
