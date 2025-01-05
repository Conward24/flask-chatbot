from openai import OpenAI

client = OpenAI(api_key="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")

# Set your OpenAI API key

def print_response(content):
    print(content)

# Make a test API call using the latest syntax
response = client.chat.completions.create(model="gpt-3.5-turbo",
messages=[
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Write a haiku about AI."}
])

# Print the AI's response
print_response(response.choices[0].message.content)
