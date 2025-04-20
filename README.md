# Spero Therapeutic Chatbot

Spero is a therapy chatbot that uses Retrieval-Augmented Generation (RAG) and Natural Language Processing (NLP) to provide emotionally supportive, contextually relevant conversations. This project evaluates the performance of two models—OpenAI’s GPT-4o and DeepSeek—in generating therapeutic responses. The comparison focuses on response quality, using ROUGE and BLEU metrics, and therapeutic relevance. By combining conversational AI with psychiatric assistance, Spero aims to enhance mental health support accessibility and help individuals manage their well-being.

## Key Features

- **Dual Model Evaluation**: Implements both the GPT-4o-mini and DeepSeek v1 models for therapeutic response generation.
- **Gradio Interface**: Provides an interactive interface for real-time testing of chatbot responses.
- **Evaluation Metrics**: Uses BLEU and ROUGE scores to assess response quality based on therapy transcripts.
- **Therapeutic Support**: Simulates a supportive conversational environment grounded in psychiatric principles.

## Dataset

This project uses the **Synthetic Therapy Conversations** dataset from Kaggle, which contains simulated therapy conversations that serve as the foundation for training and evaluating the chatbot’s performance. The dataset can be found here:

[Synthetic Therapy Conversations Dataset on Kaggle](https://www.kaggle.com/datasets/thedevastator/synthetic-therapy-conversations-dataset)

## Setup

1. **Clone the Repository**

   ```bash
   git clone https://github.com/mirzaahmadi/Spero_Therapeutic_Chatbot.git
   cd Spero_Therapeutic_Chatbot
    ```

2. **Preprocess the Data**

   Before running the chatbot interface and creating the database, preprocess the raw therapy transcripts.

   ```bash
    python preprocessing.py
   ``` 
   
4. **Ingest the Data into the ChromaDB Database**

   Once preprocessed, the data can be ingested into the ChromaDB Database.
   
  ```bash
    python ingest_database.ipynb
   ```

4. **Check the Database Size**

   Check the size of the ingested database to ensure the data was loaded correctly.

```bash
python check_size_database.py
```

5. **Run the Chatbot Interface**

   Run the Gradio Chatbot interface. This will start a local server, and you can interact with the chatbot through a web interface. 
   
```bash
python chatbot.py
```

6. **Convert Reference Transcript for Evaluation**

   The preprocessed therapy transcript requires to be formatted to be evaluated using the metrics. Choose a sample from the reference to test with the models. 
   
```bash
python convert_transcript.py
```

7. **Evaluate the Models**

   After running the chatbot interface, evaluate the models using BLEU and ROUGE metrics. The script chatbot.py shouldn't be running to recieve evaluation metrics for the models. 

```bash
python evaluate_models.py
```

## Evaluation Metrics 

The chatbot's responses are evaluated using two key metrics:

**BLEU (Bilingual Evaluation Understudy)**: A metric that compares the chatbot’s generated responses with a reference (therapy) transcript by measuring n-gram overlap. BLEU scores range from 0 to 1, where a higher score indicates a better match with the reference.
- **Interpretation**: Higher BLEU scores indicate that the generated responses closely resemble the reference transcript, which implies better linguistic similarity.

**ROUGE (Recall-Oriented Understudy for Gisting Evaluation)**: Measures the overlap between the generated response and reference response in terms of n-grams, focusing on recall rather than precision. ROUGE scores provide insight into how well the chatbot retains the key elements of the conversation.
- **Interpretation**: Higher ROUGE scores suggest better coverage and relevance of the response with respect to the reference.

## Usage 

## License
This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgements

- GPT-4o-mini: OpenAI’s GPT-4 model, a transformer-based model that excels in generating human-like text and is used for generating therapeutic responses in this project.
- DeepSeek: A deep learning model specifically designed for mental health applications, which is used to generate therapeutic responses and simulate mental health support.
- Gradio: An open-source framework for creating user interfaces with machine learning models, which is used to build the interactive interface for this project.
- Synthetic Therapy Conversations Dataset: The dataset used for training and testing the chatbot, sourced from Kaggle, providing a set of simulated therapy conversations for model evaluation.
