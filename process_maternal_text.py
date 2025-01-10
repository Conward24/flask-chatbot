import json

# Load the text file
with open("The_Prospective_Mother.txt", "r", encoding="utf-8") as file:
    text = file.read()

# Split text into chapters
chapters = text.split("CHAPTER ")
structured_data = []

for chapter in chapters[1:]:  # Skip the introduction
    lines = chapter.splitlines()
    chapter_title = lines[0].strip()  # Chapter title is the first line
    chapter_content = "\n".join(lines[1:]).strip()  # Remaining content

    # Example structure: Add chapter to JSON
    structured_data.append({
        "title": chapter_title,
        "content": chapter_content,
        "source": "The Prospective Mother, J. Morris Slemons"
    })

# Save structured data to JSON
with open("structured_maternal_guide.json", "w", encoding="utf-8") as json_file:
    json.dump(structured_data, json_file, indent=4)

print("Extraction complete! Data saved to structured_maternal_guide.json.")
