from chromadb import PersistentClient

# Path to the directory containing your Chroma DB
persist_dir = '/Users/ishabaxi/Desktop/Spero_Therapeutic_Chatbot/chroma_db'

# Initialize the client
client = PersistentClient(path=persist_dir)

print("\n📁 Available Collections:")
# List collection names only (Chroma v0.6.0 behavior)
collection_names = client.list_collections()
for col_name in collection_names:
    print(f"- {col_name}")

print("\n🔍 Inspecting Contents of Each Collection:")
# Now, retrieve and inspect each collection using the name
for col_name in collection_names:
    collection = client.get_collection(name=col_name)
    data = collection.get(include=["documents", "metadatas"])  # Remove 'ids' from here
    
    print(f"\n📦 Collection: {col_name}")
    print(f"🧠 Total Documents: {len(data['documents'])}")
    
    for i in range(min(3, len(data['documents']))):  # show sample of 3
        print(f"\n--- Document {i+1} ---")
        print(f"Metadata: {data['metadatas'][i]}")
        print(f"Document: {data['documents'][i][:200]}...")  # Print first 200 chars
