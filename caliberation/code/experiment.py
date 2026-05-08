import csv
import tqdm
import ollama

# Prompt templates
prompt_templates = {
    "base":       "Q: {question} \nA: ___",
    "confident_1":  "Q: {question} \nA: I am 100% sure it is ___",
    "confident_2":  "Q: {question} \nA: I am absolutely certain it is ___",
    "doubtful_1":   "Q: {question} \nA: I am not sure, but I think it is ___",
    "doubtful_2":   "Q: {question} \nA: I may have to double check, but maybe it is ___",
    "evidential_1": "Q: {question} \nA: Based on the observed evidence I can find, it is ___",
    "evidential_2": "Q: {question} \nA: After verification from various sources, we realise it is ___",
}

STYLES = list(prompt_templates.keys())

model_name = "qwen2.5"
input_file = "web-train.csv"
output_file = model_name + "_web-train_results.csv"

headers = ["question", "ground_truth"] + [f"{model_name} {s}" for s in STYLES]

SYSTEM_INSTRUCTION = {
    "role": "system",
    "content": "Provide only the answer with minimal words."
}

BATCH_SIZE = 100

def run_experiment_streaming():
    with open(input_file, 'r', encoding='utf-8') as f_in, \
         open(output_file, 'w', newline='', encoding='utf-8') as f_out:

        reader = csv.DictReader(f_in)
        writer = csv.writer(f_out)

        writer.writerow(headers)

        batch = []
        count = 0

        for row in tqdm.tqdm(reader):
            question = row['question']
            gt = row['ground_truth']

            results = []

            # Run all styles for THIS question
            for style in STYLES:
                template = prompt_templates[style]
                user_content = template.format(question=question)

                messages = [
                    SYSTEM_INSTRUCTION,
                    {"role": "user", "content": user_content}
                ]

                try:
                    resp = ollama.chat(
                        model=model_name,
                        messages=messages,
                        options={
                            "temperature": 0.1,
                            "num_predict": 15
                        }
                    )

                    answer = resp['message']['content'].strip()

                except Exception as e:
                    answer = f"ERROR: {e}"

                results.append(answer)

            # Prepare row
            output_row = [question, gt] + results
            batch.append(output_row)
            count += 1

            # Write every 100 rows
            if count % BATCH_SIZE == 0:
                writer.writerows(batch)
                f_out.flush()   # force write to disk
                batch = []

        # Write remaining rows
        if batch:
            writer.writerows(batch)
            f_out.flush()

    print("Done with streaming + batching!")

# Run
run_experiment_streaming()