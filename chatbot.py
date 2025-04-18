""" 
This code defines a chatbot application that uses AI models (GPT-4o or DeepSeek) to interact with users through a Gradio interface. The chatbot's primary role is to provide mental health support by responding empathetically using a knowledge base derived from documents stored in a vector database. The operator is prompted to choose between two models (GPT-4o or DeepSeek), and the chatbot fetches relevant information from the database to generate responses. It handles user input, queries the chosen model, and streams responses back to the user. The code integrates multiple components, including the use of embeddings for document search and API calls to DeepSeek, and it ensures that the chatbot remains within the scope of the provided knowledge when responding.
"""

# === LOAD IMPORTS ===
import requests
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from dotenv import load_dotenv
import os
import hashlib
import json
import gradio as gr


# === LOAD API KEYS (gpt-4o and deepseek-r1)
load_dotenv() # Load environment variables (OpenAI key)

DEEPSEEK_API_URL = "https://openrouter.ai/api/v1/chat/completions" # DeepSeek URL Key
DEEPSEEK_API_KEY = "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" # DeepSeek API Key


# === SPECIFY FILE PATHS AND EMBEDDINGS ===
CHROMA_PATH = r"Chroma_DB"
CACHE_DIR = "response_cache"
os.makedirs(CACHE_DIR, exist_ok=True)

embedding_function = OpenAIEmbeddings(model="text-embedding-3-large")


# === 'main()' FUNCTION TO PROMPT FOR MODEL CHOICE AND START CHATBOT ===
def main():
    choices = ['gpt-4o', 'deepseek-r1']
    global model_choice # Global variable indicating which model is chosen by operator
    while True:
        model_choice = input("Input model name ('gpt-4o' or 'deepseek-r1'): ").strip() # Operator inputs model of choice
        if model_choice not in choices:
            print("Invalid input! Please enter 'gpt-4o' or 'deepseek-r1'.")
        else:
            break
    global LLM  # Global variable to hold the model
    LLM = pick_model(model_choice)  # Choose model based on input

    chat_interface.launch(debug=True)  # Start Gradio UI with the chosen model
    
    
# === SPECIFY WHICH MODEL TO USE === 
def pick_model(mdl):
    """ 
    Using the specified model (Either OpenAI or DeepSeek), create the LLM
    """
    if mdl == "gpt-4o":
        return ChatOpenAI(temperature=0.5, model="gpt-4o-mini")
    elif mdl == "deepseek-r1":
        return "deepseek-r1"  # We will handle DeepSeek separately in stream_response
    else:
        raise ValueError(f"Unknown model: {mdl}")


# === STREAM RESPONSE FOR GRADIO ===
def stream_response(message, history):
    """ 
    The stream_response function is responsible for processing the user's input message, retrieving relevant information, querying the appropriate model (either OpenAI’s GPT-4o or DeepSeek), and streaming the response back to the user through the Gradio interface.
    """
    docs = get_top_docs(message)
    # The result of this line is a string that consists of the combined contents of the top k most similar documents, separated by two newlines (\n\n). This combined string (knowledge) is used as the "knowledge base" that the model can use to generate a response.
    knowledge = "\n\n".join([doc.page_content for doc in docs])

    rag_prompt = f"""
    Your name is Spero, a compassionate AI mental health assistant.  
    Your role is to support users by responding empathetically and clearly, using primarily the information provided in the “Knowledge” section.  
    You do not use external knowledge, make assumptions, or provide facts that are not explicitly stated in the knowledge.  

    However, in rare cases where it is necessary for maintaining a coherent and helpful conversation — such as understanding basic emotions, phrasing natural follow-up questions, or briefly grounding the user in shared human experience — you may draw from *general, commonly understood knowledge*.  
    You must **never use clinical facts, therapy research, or external therapeutic methods** that are not present in the “Knowledge” section.  
    Your priority is always to respond using the database provided. Use general knowledge *sparingly* and only when essential for clarity or emotional connection.

    If a user asks something clearly outside the scope of the provided knowledge, respond with:  
    "I'm sorry, I couldn’t find the answer based on the information I have."

    You are allowed to respond naturally to basic greetings, check-ins, or common phrases (e.g., "how are you?", "thank you", etc.) in a friendly and helpful tone.  

    Like a therapist, support the user by helping them develop coping mechanisms, challenge negative thought patterns, and promote self-awareness and growth.  
    Stay supportive, grounded, and gentle in your tone.  
    Never speculate or offer clinical advice.  
    Always respect the limits of your knowledge.

    ---

    To improve user experience, make your responses feel like part of a natural, two-way conversation.  
    Avoid overexplaining or giving long, lecture-style responses — speak simply and warmly, like a person would.  
    Keep your tone thoughtful and human. Respond in short, clear paragraphs.  
    Prioritize emotional resonance and connection over information-dumping.  
    Be specific and relevant — do not repeat general advice if the user is asking for something more focused.  
    Adapt to the user’s intent and emotional tone, and avoid giving the same advice multiple times.  
    Ask reflective follow-up questions only when they deepen the conversation — not as a filler.  
    Leave space for the user to process or respond, just like in a real therapeutic conversation.

    ---

    The question: {message}  
    Conversation history: {history}  
    The knowledge: {knowledge}
    """

    # Check which model to use and get the response accordingly
    if model_choice == "gpt-4o":
        response = ""
        for chunk in LLM.stream(rag_prompt):  # This works for OpenAI models with streaming ('stream' function is part of the OpenAI model). The stream method allows you to receive the model’s response in chunks rather than all at once.
            response += chunk.content
            yield response
    elif model_choice == "deepseek-r1":
        # Send the prompt to the DeepSeek API and get the response
        response = deepseek_api_call(rag_prompt)
        yield response
    else:
        raise ValueError("Unknown model name")


# === GRADIO INTERFACE ===
""" 
The 'fn' parameter is used to specify the function which handles user input and produces the output. In this case, we pass 'stream_response' to gradio.
so even though when we define the function 'stream_response' above with 'message' history' parameters (and we don't specify them specifically within this function 'stream_response'), gradio automatically provides 'message' and 'history' into the stream response function. 

Gradio handles user input = When the user types a message into the Gradio interface, it automatically captures that message as the message parameter
Gradio tracks history = Gradio keeps track of the conversation history and passes that into the stream_response function as the 'history' parameter
Gradio then invokes stream_response = After the user sends a message, Gradio calls the stream_response function and automatically passes the message and history arguments
"""
chat_interface = gr.ChatInterface(
    fn=stream_response, 
    title="Hey, I'm Spero🕊️! What's on your mind?",
    description="I'm here to support you. Type whatever's on your mind, and we’ll talk through it together 💬",
    textbox=gr.Textbox(placeholder="How are you feeling today?", container=False),
    theme=gr.themes.Soft(
        primary_hue="purple",
        secondary_hue="purple",
        neutral_hue="zinc",
        radius_size="lg",
        font=["Poppins", "Inter","sans-serif"]),
    type="messages" 
)


# === DEEPSEEK API CALL ===
def deepseek_api_call(prompt):
    """
    This function sends a request to the DeepSeek API and returns the response.
    """
    headers = {
        "Authorization": f"Bearer {DEEPSEEK_API_KEY}", # Add the authorization API key
        "Content-Type": "application/json" # Telling the server that the data we're sending is in JSON format
    }

    # Format the data (This includes the model name and the user input) to send to the API
    data = {
        "model": "deepseek/deepseek-r1:free", # specify the model name
        "messages": [{"role": "user", "content": prompt}] # The actual user input (the prompt) that will be sent to the model
    }

    # The POST request to the DeepSeek API is essentially a way to send data (specifically a prompt in this case) to the API so that the model can process the request and generate a response
    response = requests.post(DEEPSEEK_API_URL, headers=headers, json=data)

    if response.status_code == 200: # Now, check if the response status is 200 (which indicates a successful request)
        response_data = response.json() # Parse the response
        
        """ 
        response_data, which we parsed above, is expected to have the key 'choices' which contains an array of possible responses. 'get("choices"...) tries to get the value associated with 'choices'. The first item os the 'choices' list is extracted with '[0]' - this is generally the best response. Now, each item within the 'choices' array (in our case, the first array) is expected to have a "message" key, which holds the actual message content from the model. 'get("message"...) looks for this 'message' key, which holds the actual message content from the model. now the 'message' dictionary is expected to contain a 'content' key which finally contains the actual text of the model's response. The structure of the DeepSeek API might look something like this:
        
                {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": "The meaning of life is subjective and can vary based on individual beliefs, values, and experiences. Many people find meaning through relationships, personal growth, or contributing to the greater good."
                    }
                }
            ]
        }
        """
        return response_data.get("choices", [{}])[0].get("message", {}).get("content", "No response content")
    else:
        print(f"Error: {response.status_code}, {response.text}")
        return "Sorry, there was an error while processing your request."


# === SAFECHROMA CLASS - EXTENDS THE CHROMA CLASS ===
class SafeChroma(Chroma):
    """ 
    SafeChroma extends the Chroma class, which is used for handling and querying a vector database.
    """
    # similarity_search_with_score performs the similarity search. It takes a query (user input) and return the top k results.
    def similarity_search_with_score(self, query, k=5, **kwargs):
        embedded_query = embedding_function.embed_query(query) #query is converted into a vector with embedding function
        # 'self._collection.query' is the query that gets sent to the vector database, searching for most similar documents based on embedded query
        results = self._collection.query(
            query_embeddings=[embedded_query],
            n_results=k,
            include=["documents", "metadatas", "distances"]
        )
        docs_and_scores = [] # This creates an empty list to store the processed documents and their associated similarity scores.
        for doc, metadata, distance in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        ):
            if doc:
                docs_and_scores.append(
                    (Document(page_content=doc or "", metadata=metadata or {}), distance)
                )
        """ 
        eg.
        docs_and_scores = [
            (Document(page_content="Doc1 content", metadata={"id": 1}), 0.95),
            (Document(page_content="Doc2 content", metadata={"id": 2}), 0.92),
            (Document(page_content="Doc3 content", metadata={"id": 3}), 0.85)
        ]
        """
        return docs_and_scores


# Vector DB
vector_store = SafeChroma(
    collection_name="therapy_collection",
    embedding_function=embedding_function,
    persist_directory=CHROMA_PATH,
)


# === FIND THE TOP k MOST SIMILAR DOCUMENTS TO USER'S QUERY ===
def get_top_docs(message, k=5): # message is the query that the user has provided
    docs_and_scores = vector_store.similarity_search_with_score(message, k=k)
    return [doc for doc, _ in docs_and_scores]


# Launch the chatbot interface
if __name__ == "__main__":
    main()