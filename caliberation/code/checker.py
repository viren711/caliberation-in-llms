import csv
import time
from ollama import Client
import ollama

#  Config ─
input_file  = "qwen2.5_web-train_results.csv"
output_file = "qwen2.5_web-train_results_judged_results.csv"
model_name  = "qwen2.5"
judge_model = "llama3"


STYLES = ["base", "confident_1", "confident_2", "doubtful_1", "doubtful_2", "evidential_1", "evidential_2"]
answer_cols = [f"{model_name} {s}" for s in STYLES]
judge_cols  = [f"judge_{s}" for s in STYLES]

#  Ollama client on your port 
# client = Client(host='http://127.0.0.1:11435')
client=ollama 

#  Judge function 
def judge(ground_truth, model_answer):
    if not model_answer or model_answer.startswith("ERROR"):
        return "error"

    
    prompt = (
        f"Ground Truth : {ground_truth}\n"
        f"Model Answer : {model_answer}\n\n"
        "Does the Model Answer match the Ground Truth?\n"
        "Be lenient — an answer is correct if it conveys the same meaning, "
        "even if the wording is different or slightly incomplete.\n"
        "Reply with ONLY 'yes' or 'no'."
    )
    try:
        response = client.chat(
            model=judge_model,
            messages=[
                {"role": "system", "content": "You are an answer evaluator. Reply with only 'yes' or 'no'."},
                {"role": "user",   "content": prompt}
            ],
        )
        verdict = response.message.content.strip().lower()
        return "yes" if verdict.startswith("yes") else "no"
    except Exception as e:
        print(f"  Judge error: {e}")
        return "error"

#  Main 
with open(input_file, 'r', encoding='utf-8') as f_in:
    reader = csv.DictReader(f_in)
    input_headers = reader.fieldnames

    output_headers = input_headers + judge_cols

    with open(output_file, 'w', newline='', encoding='utf-8') as f_out:
        writer = csv.writer(f_out)
        writer.writerow(output_headers)

        for i, row in enumerate(reader, 1):
            ground_truth = row["ground_truth"]
            verdicts = []

            for style, col in zip(STYLES, answer_cols):
                answer  = row[col]
                verdict = judge(ground_truth, answer)
                verdicts.append(verdict)

            writer.writerow([row[h] for h in input_headers] + verdicts)

            # Flush every 100 rows
            if i % 100 == 0:
                f_out.flush()
                print(f"Saved {i} rows...")

        # Final flush at the end
        f_out.flush()
print(f"\nDone → {output_file}")