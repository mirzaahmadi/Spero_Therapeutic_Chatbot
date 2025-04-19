from chromadb import PersistentClient

# Specify the path to the directory where your Chroma DB is stored. 
persist_dir = '/Users/ishabaxi/Desktop/Spero_Therapeutic_Chatbot/chroma_db'

# Create an instance of PersistentClient to access the Chroma database at the specified location (persist_dir).
client = PersistentClient(path=persist_dir)

# Display the collections available in the Chroma DB.
print("\n📁 Available Collections:")

# Retrieve the list of collection names from the Chroma client
collection_names = client.list_collections()

# Iterate over each collection name and print it to the console
for col_name in collection_names:
    print(f"- {col_name}")

# For each collection, retrieve its documents and metadata, excluding 'ids' for brevity.
print("\n🔍 Inspecting Contents of Each Collection:")
for col_name in collection_names:
    # Retrieve the collection by its name
    collection = client.get_collection(name=col_name)
    # Get the documents and metadata for the collection
    data = collection.get(include=["documents"])  # Remove 'ids' from raw file
    
    # Print the collection name and the number of documents it contains
    print(f"\n📦 Collection: {col_name}")
    print(f"🧠 Total Documents: {len(data['documents'])}")
    
    # Show a sample of 3 documents 
    for i in range(min(3, len(data['documents']))):  
        print(f"\n--- Document {i+1} ---")
        print(f"Document: {data['documents'][i][:200]}...")  # Print first 200 chars
