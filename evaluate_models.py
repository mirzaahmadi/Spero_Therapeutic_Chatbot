import ast
import json
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer
from chatbot import get_model_response, pick_model
import chatbot  # Access global vars like model_choice, LLM

# === SETUP SCORING TOOLS ===
def setup_scoring_tools():
    """ 
    Initializes BLEU and ROUGE scoring tools. 
    Uses smoothing for BLEU scores and sets up the ROUGE scorer with the required metrics.
    """
    smoothing = SmoothingFunction().method1
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    return smoothing, scorer

smoothing, scorer = setup_scoring_tools()

# === LOAD THERAPY TRANSCRIPT ===
def load_transcript(file_path):
    """ 
    Loads and processes the therapy transcript from a JSON file.
    The file is expected to contain a list of turns with roles and text.
    """
    with open(file_path, "r") as f:
        raw_transcript = f.read()

    return ast.literal_eval(raw_transcript.replace('""', '"'))

transcript = load_transcript("therapy_transcript.json")

# === EVALUATION CONTAINERS ===
# Dictionaries to store BLEU and ROUGE scores for each model
bleu_scores = {"gpt4o": [], "deepseek": []}
rouge_scores = {
    "gpt4o": {"rouge1": [], "rouge2": [], "rougeL": []},
    "deepseek": {"rouge1": [], "rouge2": [], "rougeL": []}
}

# === LOOP THROUGH TRANSCRIPT PAIRS ===
def evaluate_responses(transcript):
    """ 
    Loops through the transcript and compares model responses with reference text using BLEU and ROUGE scores.
    Evaluates both GPT-4o and DeepSeek models for each pair of user input and therapist response.
    """
    results = []
    for i in range(len(transcript) - 1):
        if transcript[i]['role'] == 'COUNSELEE' and transcript[i + 1]['role'] == 'THERAPIST':
            user_input = transcript[i]['text']
            reference = transcript[i + 1]['text']
            print(f"\n➡️ Prompting models for: {user_input[:60]}...")

            # === GPT-4o RESPONSE ===
            chatbot.model_choice = "gpt-4o"
            chatbot.LLM = pick_model("gpt-4o")
            gpt4o_resp = get_model_response(user_input)

            # === DEEPSEEK RESPONSE ===
            chatbot.model_choice = "deepseek-r1"
            chatbot.LLM = "deepseek-r1"
            deepseek_resp = get_model_response(user_input)

            # === BLEU SCORE CALCULATION ===
            ref_tokens = [reference.split()]
            gpt_bleu = sentence_bleu(ref_tokens, gpt4o_resp.split(), smoothing_function=smoothing)
            ds_bleu = sentence_bleu(ref_tokens, deepseek_resp.split(), smoothing_function=smoothing)

            # === ROUGE SCORE CALCULATION ===
            gpt_rouge = scorer.score(reference, gpt4o_resp)
            ds_rouge = scorer.score(reference, deepseek_resp)

            # === STORE INDIVIDUAL SCORES ===
            bleu_scores["gpt4o"].append(gpt_bleu)
            bleu_scores["deepseek"].append(ds_bleu)
            for r in ["rouge1", "rouge2", "rougeL"]:
                rouge_scores["gpt4o"][r].append(gpt_rouge[r].fmeasure)
                rouge_scores["deepseek"][r].append(ds_rouge[r].fmeasure)

            # === PRINT RESULTS ===
            print(f"\nREFERENCE: {reference}")
            print(f"GPT-4o: {gpt4o_resp}")
            print(f"DeepSeek: {deepseek_resp}")
            print(f"BLEU (GPT-4o): {gpt_bleu:.4f} | BLEU (DeepSeek): {ds_bleu:.4f}")
            for r in ["rouge1", "rouge2", "rougeL"]:
                print(f"{r.upper()} F1 (GPT-4o): {gpt_rouge[r].fmeasure:.4f} | {r.upper()} F1 (DeepSeek): {ds_rouge[r].fmeasure:.4f}")

            # === SAVE RESULTS FOR ANALYSIS ===
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

    return results

results = evaluate_responses(transcript)

# === AVERAGE SCORES CALCULATION ===
def avg(lst): 
    """ 
    Calculate the average of a list of numbers, return 0 if the list is empty.
    """
    return sum(lst) / len(lst) if lst else 0

# === PRINT AVERAGE SCORES ===
def print_average_scores():
    """ 
    Prints average BLEU and ROUGE scores for both GPT-4o and DeepSeek models.
    """
    print("\n📊 AVERAGE SCORES:")
    print(f"\nBLEU (GPT-4o): {avg(bleu_scores['gpt4o']):.4f}")
    print(f"BLEU (DeepSeek): {avg(bleu_scores['deepseek']):.4f}")

    for r in ["rouge1", "rouge2", "rougeL"]:
        print(f"{r.upper()} (GPT-4o): {avg(rouge_scores['gpt4o'][r]):.4f}")
        print(f"{r.upper()} (DeepSeek): {avg(rouge_scores['deepseek'][r]):.4f}")

print_average_scores()

# === SAVE RESULTS TO JSON ===
def save_results_to_json(results, file_path="evaluation_results.json"):
    """ 
    Saves the evaluation results to a JSON file.
    """
    with open(file_path, "w") as f:
        json.dump(results, f, indent=2)

save_results_to_json(results)
