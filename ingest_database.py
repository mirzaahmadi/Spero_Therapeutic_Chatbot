""" 
This script basically just adds our PDF content to a vector database - in fact, once the database is made, we can even delete this (if we want, but we won't)
"""

#Load the necessary libraries

from langchain_community.document_loaders import CSVLoader # import document loaded for reading CSV files
from langchain_text_splitters import RecursiveCharacterTextSplitter # Imports a text splitter to break long text into chunks
from langchain_openai.embeddings import OpenAIEmbeddings # Imports OpenAI's embedding model for text vectorization
from langchain_chroma import Chroma # Imports ChromaDB, a vector database for storing embeddings - Often used in RAG
from uuid import uuid4 # Imports UUID generator for unique identifiers

from dotenv import load_dotenv # the following is required to load the .env file so that we can actually use our OPEN_AI_API key
load_dotenv()

# configuration - so the 'data' directory, which houses our CSV file, is stored in the DATA_PATH variable
DATA_PATH = r"data/test.csv"

# The 'chroma_db' vector database - NOTE: This directory has not been made yet because the script will create it itself
CHROMA_PATH = r"chroma_db"

# initiate embeddings model, which will calculate the embeddings. this model will convert the language into vectorized embeddings
embeddings_model = OpenAIEmbeddings(model="text-embedding-3-large")

# initiate the vector storage database
# This is the semantic database where we will add all of the chunks of the file
vector_store = Chroma(
    collection_name="example_collection",# A collection is a portion of a database where all related embeddings are stored - so like if working with multiple datasets, we seperate the embeddings into different collections, so that the chatbot queries certain collections depending on what it's asked
    embedding_function=embeddings_model, #This is where we use the embeddings model we specified above to calculate the embeddings
    persist_directory=CHROMA_PATH, # Directory where ChromaDB will store the embeddings for persistence
)

""" 
# loading the CSV document - this is where we actually load the CSV files
"""

# This creates a loader object that looks for CSV files in 'data', scans through it and LOADS all the CSV files it finds
loader = CSVLoader(DATA_PATH, encoding="utf-8")

# This will load all the CONTENT from the CSV dataset, returning them as a list of documents
raw_documents = loader.load()


print(f"Number of raw documents: {len(raw_documents)}")

# split the document; I can experiment with my features for the parameters below; but below just creates a text splitter object
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=300,
    chunk_overlap=100,
    length_function=len,
    is_separator_regex=False,
)

# creating the chunks - the text splitter object takes the raw text (which is in raw_documents) and splits it into chunks using the text_splitter object - the result is a list of text chunks 
chunks = text_splitter.split_documents(raw_documents)

print(f"Number of chunks: {len(chunks)}")

# creating unique ID's - with a unique identifier for every chunk we can edit and delete them - not shows in this tutorial though
uuids = [str(uuid4()) for _ in range(len(chunks))]

# adding chunks to vector database
batch_size = 1000  # Adjust batch size
for i in range(0, len(chunks), batch_size):
    batch = chunks[i : i + batch_size]
    batch_ids = uuids[i : i + batch_size]
    vector_store.add_documents(documents=batch, ids=batch_ids) # This line also assigns each chunk with a unique ID before putting it in the vector database
    print("batch", i, " finished")
    
all_documents = vector_store.get()  # Get all documents from the collection
print(f"Number of documents in the collection: {len(all_documents)}")