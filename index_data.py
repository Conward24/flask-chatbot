import json
import os  # Ensure os is imported
import warnings
from langchain_community.embeddings import OpenAIEmbeddings
from pinecone import Pinecone, ServerlessSpec

# Suppress warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", message="NotOpenSSLWarning")

# Load structured data
with open("structured_maternal_guide.json", "r", encoding="utf-8") as f:
    maternal_data = json.load(f)

# Initialize Pinecone
index_name = "maternal-knowledge"  # Replace with your index name
pinecone_instance = Pinecone(
    api_key=os.environ.get("PINECONE_API_KEY")
)

# Create index if it doesn't exist
if index_name not in pinecone_instance.index_manager.list_indexes().names():
    pinecone_instance.index_manager.create_index(
        name=index_name,
        dimension=1536,
        metric='cosine',
        spec=ServerlessSpec(
            cloud='aws',
            region=os.environ.get("PINECONE_ENVIRONMENT")
        )
    )

# Connect to the index
pinecone_index = pinecone_instance.index_manager.get_index(index_name)

# Initialize Embedding Model
embeddings = OpenAIEmbeddings(openai_api_key=os.environ.get("OPENAI_API_KEY"))

# Prepare and upsert data to Pinecone
vectors = []
for entry in maternal_data:
    vector = embeddings.embed_query(entry["content"])
    vectors.append({
        "id": entry["title"],  # Use the title as a unique ID
        "values": vector,
        "metadata": {
            "title": entry["title"],
            "content": entry["content"]
        }
    })

# Upsert vectors into Pinecone
pinecone_index.upsert(vectors)
print("Data indexed successfully!")
