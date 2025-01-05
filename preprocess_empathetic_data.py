import pandas as pd
import json

# File path to the train dataset
train_file = "empatheticdialogues/train.csv"

# Load the CSV file and handle bad rows
data = pd.read_csv(train_file, on_bad_lines='skip', low_memory=False)

# Rename columns based on the CSV structure
data.columns = [
    "conv_id", "utterance_idx", "context", "prompt", 
    "speaker_idx", "utterance", "selfeval", "tags"
]

# Replace null values in the 'tags' column with 'neutral'
data["tags"] = data["tags"].fillna("neutral")

# Select relevant columns (context, utterance, tags) and limit to 100 rows for simplicity
processed_data = data[["context", "utterance", "tags"]]


# Save as a JSON file
processed_data.to_json("empathetic_responses.json", orient="records", lines=False)

print("Processed data saved to 'empathetic_responses.json'")
