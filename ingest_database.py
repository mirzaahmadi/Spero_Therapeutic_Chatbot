"""
This script loads therapy conversation data from a CSV file,
splits it into chunks of dialogue turns, embeds the chunks using OpenAI,
and stores them in a persistent Chroma vector database.
"""

# === IMPORTS ===
import pandas as pd
import ast
from uuid import uuid4
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_chroma import Chroma

# === LOAD API KEY ===
load_dotenv()

# === CONFIGURATION ===
DATA_PATH = r"data/preprocessed_dataset.csv"  # Path to your dataset
CHROMA_PATH = r"chroma_db"  # Folder for Chroma to persist its database
CHUNK_TURNS = 4  # Number of utterances per chunk

# === LOAD DATA ===
df = pd.read_csv(DATA_PATH)

# === PARSE THE 'conversations' COLUMN ===
""" 
This takes in each row within each row of the dataset, and it returns a list of 'all_turns' so I have a list of every single utterrance, from client and therapist alike
"""
def parse_conversations(row, idx=None):
    try:
        return ast.literal_eval(row)
    except Exception as e:
        print(f"❌ Failed to parse row {idx}: {e}")
        return []

all_turns = []
for idx, row in df['conversations'].items():
    turns = parse_conversations(row, idx)
    all_turns.extend(turns)

print("ALL TURNS")
print(all_turns[:5])
print(f"\n✅ Parsed {len(all_turns)} total dialogue turns.")

# === CHUNK UTTERANCES ===
""" 
Since each 'chunk' in our case will be 4 utterrances (which we specified), the 'chunks_text' will return all the chunks, with 4 utterrances in each.
"""
def dialogue_chunker(dialogues, max_turns=CHUNK_TURNS):
    chunks = []
    for i in range(0, len(dialogues), max_turns):
        chunk = dialogues[i:i + max_turns]
        """ 
        # Format a list of dialogue turns into a readable block of text.
        # For each turn (a dictionary with 'role' and 'text'), format it as "ROLE: text"
        # Then join all turns in the chunk with newline characters to preserve the conversational flow.
        # Example:
        # COUNSELEE: I'm feeling overwhelmed.
        # THERAPIST: Tell me more about what's been going on.
        """
        chunk_text = "\n".join([f"{d['role']}: {d['text']}" for d in chunk]) 
        chunks.append(chunk_text)
    return chunks

chunk_texts = dialogue_chunker(all_turns)

print("CHUNK TEXTS")
print(chunk_texts[:5])
print(f"✅ Created {len(chunk_texts)} chunks.")

# === CONVERT TO DOCUMENT OBJECTS ===
documents = [Document(page_content=chunk) for chunk in chunk_texts]

print("DOCUMENTS")
print(documents[1:5])


# === SETUP VECTOR DATABASE ===
embeddings_model = OpenAIEmbeddings(model="text-embedding-3-large")
vector_store = Chroma(
    collection_name="example_collection",
    embedding_function=embeddings_model,
    persist_directory=CHROMA_PATH,
)

# === ADD DOCUMENTS TO VECTOR STORE ===
uuids = [str(uuid4()) for _ in range(len(documents))]
batch_size = 1000

print("\n📦 Inserting into Chroma DB...")
for i in range(0, len(documents), batch_size):
    batch = documents[i:i + batch_size]
    batch_ids = uuids[i:i + batch_size]
    vector_store.add_documents(documents=batch, ids=batch_ids)
    print(f"✅ Batch {i} to {i + batch_size} inserted.")

vector_store.persist()
print(f"\n✅ All {len(documents)} documents successfully inserted and saved to Chroma.")

# === VERIFY ===
all_docs = vector_store.get()
print(f"📊 Chroma DB now contains {len(all_docs)} documents.")