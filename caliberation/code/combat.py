import csv
import random
import ollama


# Config

INPUT_FILE  = "llama3_results_judged_results.csv"
OUTPUT_FILE = "llama3_combat_data_new.csv"

MODEL = "llama3"

N = 100
R = 10

SEED = 42


# Prompt styles

STYLES = [
    "base",
    "confident_1",
    "confident_2",
    "doubtful_1",
    "doubtful_2",
    "evidential_1",
    "evidential_2",
]

prompt_templates = {
    "base":
        "Q: {question}\nA: ___",

    "confident_1":
        "Q: {question}\nA: I am 100% sure it is ___",

    "confident_2":
        "Q: {question}\nA: I am absolutely certain it is ___",

    "doubtful_1":
        "Q: {question}\nA: I am not sure, but I think it is ___",

    "doubtful_2":
        "Q: {question}\nA: I may have to double check, but maybe it is ___",

    "evidential_1":
        "Q: {question}\nA: Based on the observed evidence I can find, it is ___",

    "evidential_2":
        "Q: {question}\nA: After verification from various sources, we realise it is ___",
}

SYSTEM_PROMPT = """
You will receive two trivia questions.

Reply with EXACTLY one line:
answer1 | answer2 | winner

Rules:
- winner must be either 1 or 2
- no explanations
- no extra text
- no notes
- no quotes
- no markdown
"""

# Ask model confidence question and parse output
def ask_confidence(question_i: str, question_j: str, style: str) -> int:

    template = prompt_templates[style]

    styled_question_i = template.format(question=question_i)
    styled_question_j = template.format(question=question_j)

    user_prompt = (
        f"(1) {styled_question_i}\n"
        f"(2) {styled_question_j}\n"
    )

    try:
        resp = ollama.chat(
            model=MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            options={
                "temperature": 0.0,
                "num_predict": 64
            }
        )


        output = resp["message"]["content"].strip()
        winner = output[-1]
        # print(output)

    except Exception as e:
        print(f"   API error: {e}")
        return -1

    # Expected:
    # (Answer_1, Answer_2, 1)

    if winner == "1":
        return 1
    elif winner == "2":
        return 2

    print(f"   Parse error. Raw output: '{output}'")
    return -1



# Load questions

def load_top_questions(path: str, n: int) -> list:

    questions = []

    with open(path, "r", encoding="utf-8") as f:

        reader = csv.DictReader(f)

        for i, row in enumerate(reader):

            if i >= n:
                break

            questions.append(row["question"])

    return questions



# Main

def main() -> None:

    questions = load_top_questions(INPUT_FILE, N)

    print(f"Loaded {len(questions)} questions from {INPUT_FILE}")

    # create output file
    with open(OUTPUT_FILE, "w", newline="", encoding="utf-8") as f:

        writer = csv.writer(f)

        writer.writerow([
            "question_i",
            "question_j",
            "winner",
            "style"
        ])

    total_rows = len(questions) * R * len(STYLES)

    print(f"Will produce {total_rows} rows -> {OUTPUT_FILE}\n")

    written = 0

    with open(OUTPUT_FILE, "a", newline="", encoding="utf-8") as f_out:

        writer = csv.writer(f_out)

        for style_idx, style in enumerate(STYLES, 1):

            print(f"[style {style_idx}/{len(STYLES)}] {style}")

            for i in range(len(questions)):

                rng = random.Random(SEED + style_idx * 1000 + i)

                candidates = [
                    k for k in range(len(questions))
                    if k != i
                ]

                sample_js = rng.sample(candidates, R)

                question_i = questions[i]

                for j in sample_js:

                    question_j = questions[j]

                    winner = ask_confidence(
                        question_i,
                        question_j,
                        style
                    )

                    writer.writerow([
                        question_i,
                        question_j,
                        winner,
                        style
                    ])

                    written += 1

                    # flush every 10 rows
                    if written % 10 == 0:
                        f_out.flush()
                        print(
                            f"   flushed {written}/{total_rows} rows"
                        )

                print(
                    f"  [{style}] q {i + 1}/{len(questions)} done "
                    f"({written}/{total_rows} rows)"
                )

        # final flush
        f_out.flush()

    print(f"\nDone! Combat data saved to -> {OUTPUT_FILE}")



# Entry
if __name__ == "__main__":
    main()