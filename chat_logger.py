import csv
import json
from pathlib import Path

CSV_FILE = Path(__file__).parent / "chat_log.csv"
FIELDNAMES = ["input", "output", "context", "token_cost", "latency"]

def log_chat(user_input, actual_output, retrieved_context, token_cost=0.0, latency=0.0):
    """Append one real chatbot interaction to chat_log.csv."""
    if isinstance(retrieved_context, str):
        context_to_save = retrieved_context
    else:
        context_to_save = json.dumps(retrieved_context or [], ensure_ascii=False)

    file_exists = CSV_FILE.exists()
    with CSV_FILE.open("a", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=FIELDNAMES)
        if not file_exists or CSV_FILE.stat().st_size == 0:
            writer.writeheader()
        writer.writerow({
            "input": user_input,
            "output": actual_output,
            "context": context_to_save,
            "token_cost": token_cost,
            "latency": latency
        })
