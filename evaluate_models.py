import ast
import json
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer
from chatbot import pick_model
import chatbot  # Access global vars like model_choice, LLM

# === SETUP SCORING TOOLS ===
def setup_scoring_tools():
    smoothing = SmoothingFunction().method1
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    return smoothing, scorer

smoothing, scorer = setup_scoring_tools()

# === LOAD THERAPY TRANSCRIPT ===
def load_transcript(file_path):
    with open(file_path, "r") as f:
        raw_transcript = f.read()
    return ast.literal_eval(raw_transcript.replace('""', '"'))

transcript = load_transcript("therapy_transcript.json")

# === EVALUATION CONTAINERS ===
bleu_scores = {"gpt4o": [], "deepseek": []}
rouge_scores = {
    "gpt4o": {"rouge1": [], "rouge2": [], "rougeL": []},
    "deepseek": {"rouge1": [], "rouge2": [], "rougeL": []}
}

# === LOOP THROUGH TRANSCRIPT PAIRS ===
def evaluate_responses(transcript):
    results = []
    for i in range(len(transcript) - 1):
        if transcript[i]['role'] == 'COUNSELEE' and transcript[i + 1]['role'] == 'THERAPIST':
            user_input = transcript[i]['text']
            reference = transcript[i + 1]['text']
            print(f"\n➡️ Prompting models for: {user_input[:60]}...")

            # === GPT-4o RESPONSE ===
            chatbot.model_choice = "gpt-4o"
            chatbot.LLM = pick_model("gpt-4o")
            gpt4o_resp = chatbot.LLM.invoke(user_input)

            # === DEEPSEEK RESPONSE ===
            chatbot.model_choice = "deepseek-r1"
            chatbot.LLM = "deepseek-r1"
            deepseek_resp = chatbot.deepseek_api_call(user_input)

            # === BLEU SCORE CALCULATION ===
            ref_tokens = [reference.split()]
            gpt_bleu = sentence_bleu(ref_tokens, gpt4o_resp.content.split(), smoothing_function=smoothing)
            ds_bleu = sentence_bleu(ref_tokens, deepseek_resp.split(), smoothing_function=smoothing)

            # === ROUGE SCORE CALCULATION ===
            gpt_rouge = scorer.score(reference, gpt4o_resp.content)
            ds_rouge = scorer.score(reference, deepseek_resp)

            # === STORE INDIVIDUAL SCORES ===
            bleu_scores["gpt4o"].append(gpt_bleu)
            bleu_scores["deepseek"].append(ds_bleu)
            for r in ["rouge1", "rouge2", "rougeL"]:
                rouge_scores["gpt4o"][r].append(gpt_rouge[r].fmeasure)
                rouge_scores["deepseek"][r].append(ds_rouge[r].fmeasure)

            # === PRINT RESULTS ===
            print(f"\nREFERENCE: {reference}")
            print(f"GPT-4o: {gpt4o_resp.content}")
            print(f"DeepSeek: {deepseek_resp}")
            print(f"BLEU (GPT-4o): {gpt_bleu:.4f} | BLEU (DeepSeek): {ds_bleu:.4f}")
            for r in ["rouge1", "rouge2", "rougeL"]:
                print(f"{r.upper()} F1 (GPT-4o): {gpt_rouge[r].fmeasure:.4f} | {r.upper()} F1 (DeepSeek): {ds_rouge[r].fmeasure:.4f}")

            # === SAVE RESULTS FOR ANALYSIS ===
            results.append({
                "user_input": user_input,
                "reference": reference,
                "gpt4o_response": gpt4o_resp.content,
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

    return results

results = evaluate_responses(transcript)

# === AVERAGE SCORES CALCULATION ===
def avg(lst): 
    return sum(lst) / len(lst) if lst else 0

# === PRINT AVERAGE SCORES ===
def print_average_scores():
    print("\n📊 AVERAGE SCORES:")
    print(f"\nBLEU (GPT-4o): {avg(bleu_scores['gpt4o']):.4f}")
    print(f"BLEU (DeepSeek): {avg(bleu_scores['deepseek']):.4f}")
    for r in ["rouge1", "rouge2", "rougeL"]:
        print(f"{r.upper()} (GPT-4o): {avg(rouge_scores['gpt4o'][r]):.4f}")
        print(f"{r.upper()} (DeepSeek): {avg(rouge_scores['deepseek'][r]):.4f}")

print_average_scores()

# === SAVE RESULTS TO JSON ===
def save_results_to_json(results, file_path="evaluation_results.json"):
    average_scores = {
        "bleu_gpt4o": avg(bleu_scores["gpt4o"]),
        "bleu_deepseek": avg(bleu_scores["deepseek"]),
        "rouge1_gpt4o": avg(rouge_scores["gpt4o"]["rouge1"]),
        "rouge1_deepseek": avg(rouge_scores["deepseek"]["rouge1"]),
        "rouge2_gpt4o": avg(rouge_scores["gpt4o"]["rouge2"]),
        "rouge2_deepseek": avg(rouge_scores["deepseek"]["rouge2"]),
        "rougeL_gpt4o": avg(rouge_scores["gpt4o"]["rougeL"]),
        "rougeL_deepseek": avg(rouge_scores["deepseek"]["rougeL"]),
    }

    output = {
        "results": results,
        "average_scores": average_scores
    }

    with open(file_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\n✅ Results saved to {file_path}")

# Call the function to save the JSON
save_results_to_json(results)
