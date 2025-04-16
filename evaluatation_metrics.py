# evaluate.py
import json
import ast
from chatbot import get_model_response
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer

# Set up scoring tools
smoothing = SmoothingFunction().method1
scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)

# Load and parse therapy transcript
with open("therapy_transcript.json", "r") as f:
    raw_transcript = f.read()

transcript = ast.literal_eval(raw_transcript.replace('""', '"'))

# Evaluate model responses
bleu_scores = []
rouge_all = {'rouge1': [], 'rouge2': [], 'rougeL': []}

for i in range(len(transcript) - 1):
    if transcript[i]['role'] == 'COUNSELEE' and transcript[i + 1]['role'] == 'THERAPIST':
        user_input = transcript[i]['text']
        reference_response = transcript[i + 1]['text']

        print(f"\n➡️ Prompting model for: {user_input[:60]}...")

        model_response = get_model_response(user_input)

        # Tokenize and compute scores
        reference_tokens = [reference_response.split()]
        model_tokens = model_response.split()

        bleu = sentence_bleu(reference_tokens, model_tokens, smoothing_function=smoothing)
        rouge_scores = scorer.score(reference_response, model_response)

        bleu_scores.append(bleu)
        for key in rouge_all:
            rouge_all[key].append(rouge_scores[key].fmeasure)

        print(f"MODEL: {model_response}")
        print(f"BLEU: {bleu:.4f}")
        for key in rouge_scores:
            print(f"{key.upper()} F1: {rouge_scores[key].fmeasure:.4f}")

# Print overall scores
print("\n📊 AVERAGE SCORES:")
print(f"BLEU: {sum(bleu_scores) / len(bleu_scores):.4f}")
for key in rouge_all:
    avg = sum(rouge_all[key]) / len(rouge_all[key])
    print(f"{key.upper()}: F1: {avg:.4f}")
