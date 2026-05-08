# Calibration in LLMs

## Abstract

Large Language Models (LLMs) achieve strong accuracy on question-answering
tasks, but accuracy alone does not capture **calibration** — how well a
model's stated confidence matches its actual correctness. A model can be
confidently wrong or uncertainly right; both are unreliable.

In this project we study whether changing the **tone** of a prompt, while
preserving its semantics, alters an LLM's accuracy and self-assessed
confidence. We design **seven prompt templates** spanning low-confidence,
high-confidence, and evidential cues, and run them on the
**[TriviaQA](https://nlp.cs.washington.edu/triviaqa/)** dataset against two
open-source models — **Qwen2.5 (0.5B)** and **LLaMA3 (8B)** served via
[Ollama](https://ollama.com/) — with additional experiments on the
proprietary **Gemini 2.5** API.

Two questions drive the pipeline:

1. **Does prompt tone affect accuracy?** — measured by an LLM-as-judge over
   answers produced under each of the 7 templates.
2. **Does prompt tone affect self-assessed confidence?** — measured by
   asking the model to pick the more-confident question in pairwise duels,
   then fitting a **Bradley–Terry–Luce (BTL)** ranking and scoring it
   against correctness with **NDCG** and top-k accuracy.

If the model is well-calibrated, the questions it ranks as "I am more
confident on this one" should also be the questions it gets right more
often, and the per-style NDCG should be high.

## Contributors

| Name | Roll Number |
|------|------|
| [Jainan Tandel](https://github.com/jainan-tandel) | 25M0368 |
| [Viren Mehta](https://github.com/viren711) | 25M2132 |
| [Ayan Seth](https://github.com/sethayan) | 25M0804 |
| [Ravi Sah](https://github.com/ravicmi) | 25D1597|

## Pipeline at a glance

The four scripts in [code/](code/) form a strict pipeline:

1. Per-question answers under **7 confidence-styled prompts**.
2. A **yes/no judgment** of each answer against the ground truth (LLM-as-judge).
3. A **pairwise "combat"** in which the model is asked to pick which of two
   trivia questions it is more confident about, repeated under each style.
4. A **BTL ranking** of the questions from those pairwise votes, evaluated
   against the binary correctness labels using **NDCG** and top-k accuracy.

---

## Repository layout

```
caliberation/
├── code/
│   ├── experiment.py     # Step 1 — generate styled answers
│   ├── checker.py        # Step 2 — LLM-as-judge yes/no grading
│   ├── combat.py         # Step 3 — pairwise confidence "combat"
│   └── btl_ndgc.py       # Step 4 — BTL fit + NDCG evaluation
├── data/
│   ├── questionAnswer15k.csv   # 15,303 trivia Q&A (primary input)
│   └── web-train87k.csv        # 87,622 trivia Q&A (large input)
└── results/
    ├── *_web-train_results*.csv          # Step 1 outputs (87k run)
    ├── *results_judged_results*.csv      # Step 2 outputs (judged)
    ├── *_combat_data_new.csv             # Step 3 outputs (pairwise votes)
    ├── *_btl_score.csv                   # Step 4: per-question BTL ranking
    └── *_btl_ndcg.csv                    # Step 4: per-style NDCG summary
```

---

## Prompt styles

All four scripts share the same set of 7 prompt templates that wrap the trivia
question. They span three calibration "tones":

| Style          | Template                                                                 |
|----------------|--------------------------------------------------------------------------|
| `base`         | `Q: {question} \nA: ___`                                                 |
| `confident_1`  | `Q: {question} \nA: I am 100% sure it is ___`                            |
| `confident_2`  | `Q: {question} \nA: I am absolutely certain it is ___`                   |
| `doubtful_1`   | `Q: {question} \nA: I am not sure, but I think it is ___`                |
| `doubtful_2`   | `Q: {question} \nA: I may have to double check, but maybe it is ___`     |
| `evidential_1` | `Q: {question} \nA: Based on the observed evidence I can find, it is ___`|
| `evidential_2` | `Q: {question} \nA: After verification from various sources, we realise it is ___` |

The same 7 styles flow through all 4 stages so the per-style scores at the end
are directly comparable.

---

## Prerequisites

Install [Ollama](https://ollama.com/download) and pull the two models used in
this project:

```bash
ollama pull llama3
ollama pull qwen2.5
```

Make sure the Ollama daemon is running (`ollama serve`) before launching any
script. Then create a Python environment and install the only two non-stdlib
dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install ollama tqdm
```

`csv`, `math`, `random`, `os`, `collections`, `typing` are all from the Python
standard library.

> **Note on file paths.** Every script reads/writes CSVs relative to the
> *current working directory* (no `argparse`). The simplest workflow is to
> `cd` into a working directory that contains the input CSV, drop the script
> beside it, and run it from there. Paths and model names are configured via
> the constants at the top of each file — edit them in place to switch
> between `llama3` / `qwen2.5` or between `questionAnswer15k.csv` /
> `web-train87k.csv`.

---

## Pipeline — run order

Run the four stages in order. Each consumes the output of the previous one.

### 1. [code/experiment.py](code/experiment.py) — generate styled answers

**What it does.** For every row of the input CSV, it asks the chosen Ollama
model to answer the question under all 7 prompt styles, with a system
instruction *"Provide only the answer with minimal words."* and decoding
options `temperature=0.1`, `num_predict=15`. Results are streamed and flushed
to disk every `BATCH_SIZE=100` rows so a long run can be safely interrupted.

**Key constants** ([code/experiment.py:18-29](code/experiment.py#L18-L29)):

- `model_name = "qwen2.5"` — Ollama model tag to call.
- `input_file = "web-train.csv"` — input CSV with columns `question, ground_truth`.
- `output_file = model_name + "_web-train_results.csv"` — generated wide CSV.
- `BATCH_SIZE = 100` — flush cadence.

**Output schema.** `question, ground_truth, {model} base, {model} confident_1,
… {model} evidential_2` (9 columns).

**How to run.**

```bash
cd caliberation/data
cp web-train87k.csv web-train.csv          # match the hardcoded name
python ../code/experiment.py                # produces qwen2.5_web-train_results.csv
```

To run it for `llama3`, just change `model_name` at the top of the file.

---

### 2. [code/checker.py](code/checker.py) — LLM-as-judge grading

**What it does.** Reads the wide answers file from step 1 and, for each of the
7 styled answers per question, asks a *judge model* (`llama3` by default)
*"Does the Model Answer match the Ground Truth?"* with lenient grading
instructions. The judge replies `yes`/`no`, which is appended as a new column
per style. Output is flushed every 100 rows.

**Key constants** ([code/checker.py:7-15](code/checker.py#L7-L15)):

- `input_file` — the `_results.csv` produced by step 1.
- `output_file` — `..._judged_results.csv` (input columns + 7 `judge_*` columns).
- `model_name` — must match the model used in step 1 (used to look up the
  answer columns `f"{model_name} {style}"`).
- `judge_model = "llama3"` — the grader. Independent of `model_name`; you can
  e.g. judge `qwen2.5` answers with `llama3` and vice-versa.

**Output schema.** All step-1 columns + `judge_base, judge_confident_1, …,
judge_evidential_2` (16 columns total).

**How to run.**

```bash
cd caliberation/data                                      # or wherever step 1 wrote to
python ../code/checker.py                                 # produces *_judged_results.csv
```

---

### 3. [code/combat.py](code/combat.py) — pairwise confidence combat

**What it does.** Loads the **first `N=100`** questions from the judged file
(treating it as the question pool). For each style and each question `i`, it
samples `R=10` other questions `j` (deterministic `random.Random(SEED + …)`),
then asks the model:

> *Reply with EXACTLY one line: `answer1 | answer2 | winner`. winner must be
> either 1 or 2. No explanations.*

The script extracts the trailing `1` or `2` as the winner. Each question pair
is presented under the same style (so the model "feels" equally confident /
doubtful about both sides). The total work is `N * R * 7 = 7,000` API calls
per model. Output is flushed every 10 rows.

**Key constants** ([code/combat.py:8-17](code/combat.py#L8-L17)):

- `INPUT_FILE = "llama3_results_judged_results.csv"` — must be the judged
  output of step 2 for the same model.
- `OUTPUT_FILE = "llama3_combat_data_new.csv"`.
- `MODEL = "llama3"`.
- `N = 100` — how many questions form the comparison pool (top of file).
- `R = 10` — how many random opponents each question fights.
- `SEED = 42` — reproducible sampling.

**Output schema.** `question_i, question_j, winner, style` where `winner ∈
{1, 2, -1}` (`-1` is a parse failure or API error).

**How to run.**

```bash
cd caliberation/data
python ../code/combat.py
```

---

### 4. [code/btl_ndgc.py](code/btl_ndgc.py) — BTL ranking + NDCG

**What it does.** This is the analysis step. For each style separately:

1. **Loads pairwise outcomes** from the combat file, dropping rows with
   `winner = -1`.

   > ⚠️ **Caveat.** The loader at [code/btl_ndgc.py:75](code/btl_ndgc.py#L75)
   > only keeps rows whose `winner` is `0` or `1`, but `combat.py` writes
   > winners as `1` or `2`. To make the BTL fit see the actual data you must
   > either (a) remap `winner` from `{1,2}` to `{1,0}` (1 = `q_i` wins,
   > 0 = `q_j` wins) before running, or (b) change the filter to accept
   > `(1, 2)` and treat `2` as `q_j` win. Without this, `n_pairs` will be 0
   > and BTL will degenerate.

2. **Fits a Bradley–Terry–Luce model** by Zermelo / Minorization-Maximization:

   ```
   P(i beats j) = exp(beta_i) / (exp(beta_i) + exp(beta_j))
   p_i  <-  W_i / sum_{j!=i} N_ij / (p_i + p_j)
   ```

   Iterates up to `MAX_ITER=10000` with L1 tolerance `TOL=1e-9`, normalizes
   strengths to a simplex, and reports `beta = log(p)`.

3. **Ranks** the 100 questions in descending BTL strength. Rank 1 = the
   question the model *says* it is most confident about (under that style).

4. **Joins** with the judged file to attach the `answer` and binary
   correctness `result` (`yes`→1, `no`→0). An `assert` enforces that the
   first 100 rows of the judged file align with the question pool used for
   combat.

5. **Computes NDCG** of the BTL ordering against the binary correctness
   labels, full as well as `@10/@25/@50`, plus naive top-k accuracy.

**Key constants** ([code/btl_ndgc.py:39-58](code/btl_ndgc.py#L39-L58)):

- `COMBAT_FILE`, `JUDGED_FILE` — inputs for step 4 (defaults are written for
  `qwen2.5`; rename to point at the `llama3` files when running for that model).
- `OUT_RANK = "qwen2.5_btl_score.csv"` — 700 rows (100 questions × 7 styles).
- `OUT_NDCG = "qwen2.5_btl_ndcg.csv"` — 7 rows, one per style.
- `MAX_ITER = 10000`, `TOL = 1e-9` — BTL MLE iteration controls.

**Output schemas.**

`*_btl_score.csv`: `question, style, btl_rank, btl_strength, btl_beta, answer,
result`.

`*_btl_ndcg.csv`: `style, n_items, n_pairs, ndcg, ndcg_at_10, ndcg_at_25,
ndcg_at_50, accuracy_top_10, accuracy_top_25, accuracy_overall`.

**How to run.**

```bash
cd caliberation/results
python ../code/btl_ndgc.py
```

The script also pretty-prints the per-style NDCG table to stdout.

---

## Data

Both files are derived from the **[TriviaQA](https://nlp.cs.washington.edu/triviaqa/)**
dataset (Joshi et al., 2017) — short factual questions paired with a
canonical short answer.

| File | Rows | Columns | Description |
|------|------|---------|-------------|
| [data/questionAnswer15k.csv](data/questionAnswer15k.csv) | 15,303 | `question, ground_truth` | TriviaQA Q&A used for the smaller end-to-end runs (the judged file in `results/` has 15,367 rows, matching this dataset). |
| [data/web-train87k.csv](data/web-train87k.csv) | 87,622 | `question, ground_truth` | Larger TriviaQA `web-train` split, used for the `qwen2.5_web-train_*` runs. |

Both files share the same two-column schema, which is the only schema
[code/experiment.py](code/experiment.py) expects.

---

## Results

All result CSVs in [results/](results/) follow the file-name convention
`{model}_{stage}.csv`. The full set of pre-computed artifacts:

### Stage 1 — styled answers

| File | Rows | Notes |
|------|------|-------|
| [results/qwen2.5_web-train_results.csv](results/qwen2.5_web-train_results.csv) | 88,146 | Qwen2.5 answers on `web-train87k.csv` under all 7 styles. |

### Stage 2 — LLM-judged correctness

| File | Rows | Notes |
|------|------|-------|
| [results/qwen2.5_web-train_results_judged_results.csv](results/qwen2.5_web-train_results_judged_results.csv) | 88,146 | Qwen2.5 answers + `judge_*` verdicts on the 87k pool. |
| [results/qwen2.5results_judged_results.csv](results/qwen2.5results_judged_results.csv) | 15,368 | Qwen2.5 answers + verdicts on the 15k pool (input to BTL for Qwen). |
| [results/llama3llama3results_judged_results_trim.csv](results/llama3llama3results_judged_results_trim.csv) | 738 | Llama3 answers + verdicts (trimmed to the question pool used for combat). |

### Stage 3 — pairwise combat

| File | Rows | Notes |
|------|------|-------|
| [results/llama3_combat_data_new.csv](results/llama3_combat_data_new.csv) | 13,201 | Llama3 pairwise winners under 7 styles. |
| [results/qwen_2.5_combat_data_new.csv](results/qwen_2.5_combat_data_new.csv) | 7,001 | Qwen2.5 pairwise winners (`100 × 10 × 7 = 7,000` + header). |

### Stage 4 — BTL rankings and NDCG

| File | Rows | Notes |
|------|------|-------|
| [results/llama3_btl_score.csv](results/llama3_btl_score.csv) | 701 | Llama3 per-question BTL rank, strength, answer, correctness (100 × 7 styles). |
| [results/qwen2.5_btl_score.csv](results/qwen2.5_btl_score.csv) | 701 | Qwen2.5 per-question BTL ranking. |
| [results/qwen_btl_score.csv](results/qwen_btl_score.csv) | 701 | Earlier Qwen BTL ranking (legacy / second run). |
| [results/llama3_btl_ndcg.csv](results/llama3_btl_ndcg.csv) | 8 | Llama3 NDCG per style (header + 7 styles). |
| [results/qwen2.5_btl_ndcg.csv](results/qwen2.5_btl_ndcg.csv) | 8 | Qwen2.5 NDCG per style. |
| [results/qwen_btl_ndcg.csv](results/qwen_btl_ndcg.csv) | 8 | Earlier Qwen NDCG (legacy). |

The `*_btl_ndcg.csv` files are the bottom-line calibration metric: a high
NDCG means the BTL ordering induced by stylized self-confidence is
well-aligned with the judge's correctness labels — i.e. the model's
*expressed* confidence tracks its *actual* accuracy on that style.

---

## End-to-end recipe (single model)

To reproduce the pipeline for, say, `qwen2.5` on `questionAnswer15k.csv`:

```bash
# 0. Prereqs
ollama serve &
ollama pull qwen2.5
ollama pull llama3        # for the judge in step 2

# 1. Styled answers
cd caliberation/data
cp questionAnswer15k.csv web-train.csv
# (edit experiment.py: model_name="qwen2.5", input_file="web-train.csv")
python ../code/experiment.py

# 2. LLM-as-judge grading
# (edit checker.py: input_file=<step 1 output>, model_name="qwen2.5", judge_model="llama3")
python ../code/checker.py

# 3. Pairwise confidence combat (top 100 questions × 10 opponents × 7 styles)
# (edit combat.py: INPUT_FILE=<step 2 output>, MODEL="qwen2.5")
python ../code/combat.py

# 4. BTL ranking + NDCG
# (edit btl_ndgc.py: COMBAT_FILE / JUDGED_FILE to match steps 3 & 2)
# Remember to remap winner {1,2} -> {1,0} before this step (see caveat above).
python ../code/btl_ndgc.py
```

Output of step 4 is the per-style NDCG table — the headline calibration
result for that model.
