from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma
import gradio as gr

# import the .env file - again we need this for the API key
from dotenv import load_dotenv
load_dotenv()

# configuration - again these just provide the relative paths to the raw data and the Chroma database
DATA_PATH = r"data"
CHROMA_PATH = r"chroma_db"

embeddings_model = OpenAIEmbeddings(model="text-embedding-3-large")

# initiate the model - NOTE: the temperature here denotes like the 'personality' in a way - the more you crank it up, you get more creative and unpredictable responses - 'extra'
LLM = ChatOpenAI(temperature=0.5, model='gpt-4o-mini')

# connect to the chromadb
vector_store = Chroma(
    collection_name="example_collection",
    embedding_function=embeddings_model,
    persist_directory=CHROMA_PATH, 
)

# Set up the vectorstore to be the retriever - here, you are fetching the top 5 results - remember, you can change this. Just note that as you increase this number, you risk choosing some options that maybe aren't as relative to the prompt as you want
num_results = 5
# Create a retriever object that actually fetches the data for you 
retriever = vector_store.as_retriever(search_kwargs={'k': num_results})

# call this function for every message added to the chatbot
def stream_response(message, history):
    print(f"Input: {message}. History: {history}\n")

    # retrieve the relevant chunks based on the question asked
    docs = retriever.invoke(message)

    # add all the chunks to 'knowledge' - this stored all the retrieved knowledge
    knowledge = ""

    # Here, we loop through all the chunks that we receive, this can be up to five chunks (if our num_results variable is 5)
    # This returns all the page content from each document and puts it into knowledge
    for doc in docs:
        knowledge += doc.page_content+"\n\n"


    # make the call to the LLM (including prompt)
    if message is not None:

        partial_message = ""

        # In the below prompt, we provide the LLM with the message from the user, but also from the conversation history so the history is kept track of - and therefore, it is aware of the entire context. The knwledge is coming from the intial retrieval and storage of relevant context from the database
        rag_prompt = f"""
        You are an assistant which answers questions based on knowledge which is provided to you.
        While answering, you don't use your internal knowledge, 
        but solely the information in the "The knowledge" section.
        You don't mention anything to the user about the provided knowledge.

        The question: {message}

        Conversation history: {history}

        The knowledge: {knowledge}

        """

        #print(rag_prompt)

        # stream the response to the Gradio App
        # The LLM.stream(rag_prompt) is used to send the prompt to the language model, and instead of waiting for the entire response to be generated before sending it to the user, it streams it in chunks. Each chunk is added to partial_message, and the yield partial_message sends the latest version of the response to the Gradio app.
        # With partial chunks being generated - the user can like actively see the reply written to them
        for response in LLM.stream(rag_prompt):
            partial_message += response.content
            yield partial_message

# initiate the Gradio app
# Here we are referring to the function 'stream_response' - this is the function that handles generating responses from the LLM
#The 'textbox' literally creates a little textbox where you type your messages, and it behaves like a smooth, auto-scrolling chat field.
chatbot = gr.ChatInterface(stream_response, textbox=gr.Textbox(placeholder="Send to the LLM...",
    container=False,
    autoscroll=True,
    scale=7),
)

# launch the Gradio app
chatbot.launch()