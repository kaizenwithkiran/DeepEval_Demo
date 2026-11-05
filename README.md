<<<<<<< HEAD
# RAG Notes Chatbot

A retrieval-augmented chatbot that answers questions using your own `.md`/`.txt` notes.
Python + LangChain + Chroma for retrieval, OpenAI for embeddings and chat, Streamlit for
the UI — with a **DeepEval evaluation suite** in `evals/` and `tests/`.

The knowledge base in `data/` is a fictional training academy: four courses, fees,
refund and transfer policy, batch schedule, certification and placement, support — plus
one note on LLM evaluation itself, so the corpus has real conceptual content and not
only lookup facts.

## Setup

1. Create a virtual environment and install dependencies:

   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```

2. Add your OpenAI API key:

   ```bash
   cp .env.example .env
   ```

   Then edit `.env` and set `OPENAI_API_KEY`. Do not quote the value — an unterminated
   quote makes `python-dotenv` print a parse warning.

3. Add your notes (`.md` or `.txt` files) to the `data/` folder. Ten notes are already
   there so you can test the app right away.

4. Build the index:

   ```bash
   python -m src.ingest
   ```

5. Run the app:

   ```bash
   streamlit run app.py
   ```

## Updating the knowledge base

Add or edit files in `data/`, then either re-run `python -m src.ingest` or click
**Re-index notes** in the app's sidebar.

## Evaluating it

Five metrics — faithfulness, relevancy, correctness, toxicity, latency:

```bash
python -m evals.run_all          # the whole suite plus a scorecard (~4.5 min)
pytest tests/ -v                 # the same thresholds, as a CI gate (~3 min)
pytest tests/ -v -m "not slow"   # deterministic guards only — 4 seconds, no API calls
```

Every run is archived locally under `results/<run-id>/` — one
`test_run_*.json` per suite plus `latency.json`, all sharing a run id so a whole
`run_all` is one comparable run. Name a run yourself with `EVAL_RUN_ID`:

```bash
EVAL_RUN_ID=baseline python -m evals.run_all
EVAL_RUN_ID=chunk_500 python -m evals.run_all
```

Export every run that has ever been archived to CSV:

```bash
python -m evals.export_results
```

That writes `results/export/`: `runs.csv` (one row per suite run, with the config
that produced it), `metrics.csv` (avg/min/max and pass counts per metric),
`cases.csv` (every case, its score and the judge's reason), `latency.csv`
(percentiles per stage) and `all_runs.json` (everything, unflattened). DeepEval
on its own only keeps a rolling snapshot of the *last* evaluate() call in
`.deepeval/`, so without this the history is gone on the next run — the exporter
folds that snapshot in too, deduplicated, so nothing already on disk is lost.
ag_full_evaluation_demo_Deepeval/rag_full_evaluation_demo$ git pull --no-rebase origin main --allow-unrelated-histories
From https://github.com/kaizenwithkiran/DeepEval_Demo
 * branch            main       -> FETCH_HEAD
Auto-merging README.md
CONFLICT (add/add): Merge conflict in README.md
Automatic merge failed; fix conflicts and then commit the result.
kiransahu@mfp-lptp-042:~/Documents/Personal_Docs/Training/10thAugBatch/Week2/W2D4/Test_Suite_2RAG/rag_full_evaluation_demo_Deepeval/rag_full_evaluation_demo$ 

**See [EVALUATIONS.md](EVALUATIONS.md)** for what each metric measures, why the
thresholds are where they are, how to read a failure, and a set of exercises that break
the app on purpose so the numbers move.

## Project layout

```
app.py                 Streamlit chat UI
src/config.py          Env vars and settings
src/ingest.py          Loads notes, chunks them, embeds, stores in Chroma
src/rag_chain.py       Retrieval + prompt + LLM call; also generate() / generate_stream()
src/rag_pipeline.py    The eval-facing wrapper — exposes the retrieved context
data/                  Your .md/.txt notes go here
goldens/               Golden datasets (the answer keys) for the eval suite
evals/                 One script per metric family, plus run_all.py
evals/export_results.py  Flattens every archived run into results/export/*.csv
tests/                 The same metrics as pytest assertions, for CI
chroma_db/             Persisted vector store (auto-created, gitignored)
results/               One folder per run, plus the CSV exports (gitignored)
```
=======
# DeepEval_Demo
>>>>>>> 269d2fe59f0d26211901191e2cc2cf52a5e6fc0a
