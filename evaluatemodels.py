import ast
import json
from chatbot import get_model_response
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer

# Setup the scoring tools
smoothing = SmoothingFunction().method1
scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)

# Load therapy transcript
with open("therapy_transcript.json", "r") as f:
    raw_transcript = f.read()

transcript = ast.literal_eval(raw_transcript.replace('""', '"'))

# Evaluation containers
results = []
bleu_scores = {"gpt4o": [], "deepseek": []}
rouge_scores = {"gpt4o": {"rouge1": [], "rouge2": [], "rougeL": []},
                "deepseek": {"rouge1": [], "rouge2": [], "rougeL": []}}

# Loop through pairs of Counselee → Therapist
for i in range(len(transcript) - 1):
    if transcript[i]['role'] == 'COUNSELEE' and transcript[i + 1]['role'] == 'THERAPIST':
        user_input = transcript[i]['text']
        reference = transcript[i + 1]['text']
        print(f"\n➡️ Prompting models for: {user_input[:60]}...")

        # Get responses
        gpt4o_resp = get_model_response(user_input, model="gpt4o")
        deepseek_resp = get_model_response(user_input, model="deepseek")

        # BLEU
        ref_tokens = [reference.split()]
        gpt_bleu = sentence_bleu(ref_tokens, gpt4o_resp.split(), smoothing_function=smoothing)
        ds_bleu = sentence_bleu(ref_tokens, deepseek_resp.split(), smoothing_function=smoothing)

        # ROUGE
        gpt_rouge = scorer.score(reference, gpt4o_resp)
        ds_rouge = scorer.score(reference, deepseek_resp)

        # Save individual scores
        bleu_scores["gpt4o"].append(gpt_bleu)
        bleu_scores["deepseek"].append(ds_bleu)
        for r in ["rouge1", "rouge2", "rougeL"]:
            rouge_scores["gpt4o"][r].append(gpt_rouge[r].fmeasure)
            rouge_scores["deepseek"][r].append(ds_rouge[r].fmeasure)

        # Print
        print(f"\nREFERENCE: {reference}")
        print(f"GPT-4o: {gpt4o_resp}")
        print(f"DeepSeek: {deepseek_resp}")
        print(f"BLEU (GPT-4o): {gpt_bleu:.4f} | BLEU (DeepSeek): {ds_bleu:.4f}")
        for r in ["rouge1", "rouge2", "rougeL"]:
            print(f"{r.upper()} F1 (GPT-4o): {gpt_rouge[r].fmeasure:.4f} | {r.upper()} F1 (DeepSeek): {ds_rouge[r].fmeasure:.4f}")

        results.append({
            "user_input": user_input,
            "reference": reference,
            "gpt4o_response": gpt4o_resp,
            "deepseek_response": deepseek_resp,
            "bleu_gpt4o": gpt_bleu,
            "bleu_deepseek": ds_bleu,
            "rouge1_gpt4o": gpt_rouge["rouge1"].fmeasure,
            "rouge1_deepseek": ds_rouge["rouge1"].fmeasure,
            "rouge2_gpt4o": gpt_rouge["rouge2"].fmeasure,
            "rouge2_deepseek": ds_rouge["rouge2"].fmeasure,
            "rougeL_gpt4o": gpt_rouge["rougeL"].fmeasure,
            "rougeL_deepseek": ds_rouge["rougeL"].fmeasure,
        })

# Print averages
print("\n📊 AVERAGE SCORES:")

def avg(lst): return sum(lst) / len(lst) if lst else 0

print(f"\nBLEU (GPT-4o): {avg(bleu_scores['gpt4o']):.4f}")
print(f"BLEU (DeepSeek): {avg(bleu_scores['deepseek']):.4f}")

for r in ["rouge1", "rouge2", "rougeL"]:
    print(f"{r.upper()} (GPT-4o): {avg(rouge_scores['gpt4o'][r]):.4f}")
    print(f"{r.upper()} (DeepSeek): {avg(rouge_scores['deepseek'][r]):.4f}")

# Save results to JSON (optional)
with open("evaluation_results.json", "w") as f:
    json.dump(results, f, indent=2)
