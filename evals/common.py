"""Shared file-loading helpers for the DeepEval demo scripts."""

import csv
import json
from pathlib import Path

from dotenv import load_dotenv

# Project root is the parent of the evals folder.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
CHAT_LOG_PATH = PROJECT_ROOT / "chat_log.csv"
GOLDENS_PATH = PROJECT_ROOT / "goldens" / "goldens.json"

# Load the project's .env file regardless of the directory used to launch Python.
load_dotenv(PROJECT_ROOT / ".env")


def load_chat_log(required_columns=("input", "output")):
    """Read chat_log.csv and check that the required columns are present."""
    if not CHAT_LOG_PATH.exists():
        raise FileNotFoundError(
            f"Could not find {CHAT_LOG_PATH}. Run the script from the project "
            "that contains chat_log.csv, or place chat_log.csv in the project root."
        )

    with open(CHAT_LOG_PATH, encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)
        columns = reader.fieldnames or []
        missing = [column for column in required_columns if column not in columns]

        if missing:
            raise ValueError(
                "chat_log.csv is missing required column(s): " + ", ".join(missing)
            )

        rows = [row for row in reader if (row.get("input") or "").strip()]

    if not rows:
        raise ValueError(f"No chat records were found in {CHAT_LOG_PATH}.")

    return rows


def load_goldens():
    """Load the preferred Golden file, with compatibility for older demo files."""
    candidate_paths = [
        GOLDENS_PATH,
        PROJECT_ROOT / "goldens" / "faithfulness.json",
        PROJECT_ROOT / "goldens" / "goldens.json",
    ]
    golden_path = next((path for path in candidate_paths if path.exists()), None)

    if golden_path is None:
        raise FileNotFoundError(
            "Could not find a Golden dataset. Create "
            f"{GOLDENS_PATH.relative_to(PROJECT_ROOT)} first."
        )

    with open(golden_path, encoding="utf-8") as file:
        goldens = json.load(file)

    if not isinstance(goldens, list) or not goldens:
        raise ValueError(f"{golden_path} must contain a non-empty JSON list.")

    seen_ids = set()
    for position, golden in enumerate(goldens, start=1):
        if not isinstance(golden, dict):
            raise ValueError(f"Golden item {position} must be a JSON object.")
        if not (golden.get("input") or golden.get("question")):
            raise ValueError(
                f"Golden item {position} needs an 'input' or 'question' field."
            )
        golden_id = str(golden.get("id", "")).strip()
        if golden_id and golden_id in seen_ids:
            raise ValueError(f"Duplicate Golden id found: {golden_id}")
        if golden_id:
            seen_ids.add(golden_id)

    print(f"Goldens loaded: {len(goldens)} ({golden_path.relative_to(PROJECT_ROOT)})")
    return goldens


def context_to_list(value):
    """Convert a context string or JSON list into DeepEval's list[str] format."""
    if value is None:
        return []

    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]

    text = str(value).strip()
    if not text:
        return []

    # Allow a CSV/JSON field that contains a serialized list of context chunks.
    if text.startswith("["):
        try:
            decoded = json.loads(text)
            if isinstance(decoded, list):
                return [str(item).strip() for item in decoded if str(item).strip()]
        except json.JSONDecodeError:
            pass

    # A normal multiline context from the RAG app is treated as one retrieved chunk.
    return [text]


def get_expected_output(golden):
    """Support the current schema and the older Synthesizer field name."""
    return (golden.get("expected_output") or golden.get("ideal_answer") or "").strip()

def _normalize_input(value):
    text = " ".join(str(value or "").casefold().split())

    # Remove quotation marks accidentally added around questions
    return text.strip().strip("\"'“”‘’").strip()

def match_chat_rows_to_goldens(chat_rows, goldens):
    """Match by id when available; otherwise match normalized input/question text."""
    goldens_by_id = {}
    goldens_by_input = {}

    for golden in goldens:
        golden_id = str(golden.get("id", "")).strip()
        golden_input = golden.get("input") or golden.get("question")

        if golden_id:
            goldens_by_id[golden_id] = golden
        key = _normalize_input(golden_input)
        goldens_by_input.setdefault(key, []).append(golden)

    matched = []
    unmatched = []
    ambiguous = []

    for row in chat_rows:
        row_id = str(row.get("id", "")).strip()
        golden = goldens_by_id.get(row_id) if row_id else None

        if golden is None:
            matches = goldens_by_input.get(_normalize_input(row.get("input")), [])
            if len(matches) == 1:
                golden = matches[0]
            elif len(matches) > 1:
                ambiguous.append(row.get("input", ""))
                continue

        if golden is None:
            unmatched.append(row.get("input", ""))
            continue

        matched.append((row, golden))

    print(f"Chats matched to Goldens: {len(matched)}")
    if unmatched:
        print(f"Chats without a Golden: {len(unmatched)}")
        for question in unmatched:
            print("  No Golden for:", question)
    if ambiguous:
        print(f"Chats with ambiguous Golden matches: {len(ambiguous)}")
        for question in ambiguous:
            print("  Add a unique id to resolve:", question)

    return matched
