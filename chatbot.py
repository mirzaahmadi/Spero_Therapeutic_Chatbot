# chatbot.py
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from dotenv import load_dotenv
import os
import hashlib
import json
import gradio as gr

# Load environment variables (OpenAI key)
load_dotenv()

# Paths
CHROMA_PATH = r"chroma_db"
CACHE_DIR = "response_cache"
os.makedirs(CACHE_DIR, exist_ok=True)

# Embeddings and model
embedding_function = OpenAIEmbeddings(model="text-embedding-3-large")
LLM = ChatOpenAI(temperature=0.5, model='gpt-4o-mini')

# Chroma wrapper
class SafeChroma(Chroma):
    def similarity_search_with_score(self, query, k=5, **kwargs):
        embedded_query = embedding_function.embed_query(query)
        results = self._collection.query(
            query_embeddings=[embedded_query],
            n_results=k,
            include=["documents", "metadatas", "distances"]
        )
        docs_and_scores = []
        for doc, metadata, distance in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        ):
            if doc:
                docs_and_scores.append(
                    (Document(page_content=doc or "", metadata=metadata or {}), distance)
                )
        return docs_and_scores

# Vector DB
vector_store = SafeChroma(
    collection_name="example_collection",
    embedding_function=embedding_function,
    persist_directory=CHROMA_PATH,
)

# Get top docs
def get_top_docs(message, k=5):
    docs_and_scores = vector_store.similarity_search_with_score(message, k=k)
    return [doc for doc, _ in docs_and_scores]

# Caching
def cache_path(message):
    key = hashlib.sha256(message.encode()).hexdigest()
    return os.path.join(CACHE_DIR, f"{key}.json")

def get_cached_response(message):
    path = cache_path(message)
    if os.path.exists(path):
        with open(path, "r") as f:
            return json.load(f)["response"]
    return None

def save_cached_response(message, response):
    with open(cache_path(message), "w") as f:
        json.dump({"response": response}, f)

# Streaming response for Gradio
def stream_response(message, history):
    docs = get_top_docs(message)
    knowledge = "\n\n".join([doc.page_content for doc in docs])

    rag_prompt = f"""
    Your name is Spero, a compassionate AI mental health assistant. 
    Your role is to support users by responding empathetically and clearly, using only the information provided in the “Knowledge” section. 
    You do not use external knowledge, make assumptions, or provide facts that are not explicitly stated in the knowledge. 
    If a user asks something outside the scope of the provided knowledge, respond with:
    "I'm sorry, I couldn’t find the answer based on the information I have."

    However, you are allowed to respond naturally to basic greetings, check-ins, or common phrases (e.g., "how are you?", "thank you", etc.) in a friendly and helpful tone.

    Stay supportive, grounded, and gentle in your tone. 
    Never speculate or offer clinical advice. 
    Always respect the limits of your knowledge.

    The question: {message}

    Conversation history: {history}

    The knowledge: {knowledge}
    """
    response = ""
    for chunk in LLM.stream(rag_prompt):
        response += chunk.content
        yield response

# Gradio interface (optional)
chat_interface = gr.ChatInterface(
    fn=stream_response,
    title="Spero - Mental Health Chatbot",
    textbox=gr.Textbox(placeholder="How are you feeling today?", container=False),
    theme="soft",
)

# 🧠 Used for automated evaluation (e.g., BLEU/ROUGE)
def get_model_response(message):
    cached = get_cached_response(message)
    if cached:
        return cached

    docs = get_top_docs(message)
    knowledge = "\n\n".join([doc.page_content for doc in docs])

    rag_prompt = f"""
    Your name is Spero, a compassionate AI mental health assistant. 
    Your role is to support users by responding empathetically and clearly, using only the information provided in the “Knowledge” section. 
    You do not use external knowledge, make assumptions, or provide facts that are not explicitly stated in the knowledge. 
    If a user asks something outside the scope of the provided knowledge, respond with:
    "I'm sorry, I couldn’t find the answer based on the information I have."

    However, you are allowed to respond naturally to basic greetings, check-ins, or common phrases (e.g., "how are you?", "thank you", etc.) in a friendly and helpful tone.

    Stay supportive, grounded, and gentle in your tone. 
    Never speculate or offer clinical advice. 
    Always respect the limits of your knowledge.

    The question: {message}
    The knowledge: {knowledge}
    """

    response = LLM.invoke(rag_prompt).content
    save_cached_response(message, response)
    return response

# Launch chatbot UI (only if you want to test manually)
if __name__ == "__main__":
    chat_interface.launch()
