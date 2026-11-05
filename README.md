# RAG Notes Chatbot with DeepEval Evaluation

A retrieval-augmented (RAG) chatbot that answers questions using your own `.md`/`.txt` notes,
with a **DeepEval evaluation suite** that scores its answers.

Python + LangChain + Chroma for retrieval, OpenAI for embeddings and chat, Streamlit for the UI.

The knowledge base in `data/` is a fictional training academy: four courses, fees, refund and
transfer policy, batch schedule, certification and placement, support — plus one note on LLM
evaluation itself, so the corpus has real conceptual content and not only lookup facts.

## Evaluation flow

```
Golden dataset (goldens/goldens.json)
        ↓
Run the chatbot (chat_app.py)  →  Execution log (chat_log.csv)
        ↓
Data preparation (evals/common.py)  →  DeepEval test cases
        ↓
Five metrics (evals/run_all.py)  →  Judge LLM (gpt-4.1-mini)
        ↓
Score + reason for every test case
```

## Prerequisites

- Python 3.10 or newer
- An OpenAI API key

## Setup

### 1. Get the code

```bash
git clone https://github.com/kaizenwithkiran/DeepEval_Demo.git
cd DeepEval_Demo
```

### 2. Create and activate a virtual environment

Mac / Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Windows:

```bash
python -m venv .venv
.venv\Scripts\activate
```

You should see `(.venv)` at the start of your terminal line.

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Add your OpenAI API key

Mac / Linux: `cp .env.example .env`
Windows: `copy .env.example .env`

Open `.env` and set your key (no quotes around the value):

```
OPENAI_API_KEY=sk-your-key-here
```

Never commit `.env` — it is already listed in `.gitignore`.

### 5. Build the knowledge base

```bash
python -m src.ingest
```

This reads the notes in `data/`, splits them into chunks, embeds them, and stores them in
`chroma_db/`. You should see a line like `Indexed N chunks from 10 documents`.

To change the knowledge base later, add or edit files in `data/` and run this command again
(or click **Re-index notes** in the app sidebar).

## Running the evaluation

### 6. Chat with the bot to create the execution log

```bash
streamlit run chat_app.py
```

- Ask the questions from `goldens/goldens.json`, copying each `input` **exactly** — the
  Hallucination and Completeness metrics match your chat to a golden by its question text.
- Every answer is saved automatically to `chat_log.csv` (input, output, retrieved context,
  cost, latency).
- Press **Ctrl+C** in the terminal to stop the app.

A sample `chat_log.csv` is included, so you can skip this step and go straight to step 7.
To start fresh, delete `chat_log.csv` — the app creates a new one.

### 7. Run all five metrics

Run this from the **project root folder** (not from inside `evals/`):

```bash
python evals/run_all.py
```

| Metric | What it compares against | Needs goldens? |
|---|---|---|
| Faithfulness | The context the chatbot actually retrieved | No |
| Answer Relevancy | Input + output | No |
| Toxicity | Output only | No |
| Hallucination | Golden context in `goldens.json` | Yes |
| Completeness | Golden expected output in `goldens.json` | Yes |

### 8. Read the results

For each test case DeepEval prints a **score** and a **reason** per metric, then a pass/fail
summary. The run ends with `EVALUATION RUN SUMMARY`.

## Troubleshooting

| Error | Fix |
|---|---|
| `OPENAI_API_KEY is not set` | `.env` is missing or the key line is wrong |
| `No module named ...` | The virtual environment is not active — repeat step 2 |
| `Could not find chat_log.csv` | You ran the command from inside `evals/` — go back to the project root |
| `Chats without a Golden` | The question typed in the app doesn't exactly match an `input` in `goldens.json` |

## Project layout

```
chat_app.py            Streamlit chat UI (evaluation mode) — logs every answer to chat_log.csv
app.py                 Simple Streamlit chat UI without logging
chat_logger.py         Helper to log chats from another bot into chat_log.csv
bot_integration_example.py  Example of using chat_logger.py
chat_log.csv           Execution log (sample included)
src/config.py          Env vars and settings
src/ingest.py          Loads notes, chunks them, embeds, stores in Chroma
src/rag_chain.py       Retrieval + prompt + LLM call
src/rag_pipeline.py    Eval-facing wrapper — returns answer, retrieved context, cost, latency
data/                  Knowledge base notes (.md/.txt)
goldens/goldens.json   Golden dataset — input, expected output, golden context
evals/common.py        Loads chat_log.csv and goldens, matches chats to goldens
evals/*.py             One script per metric
evals/run_all.py       Runs all five metrics in sequence
chroma_db/             Vector store (created by ingest, gitignored)
```
