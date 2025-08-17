import pandas as pd
import requests
from tqdm import tqdm
from pathlib import Path
import ast

# file‑paths
BASE_DIR = Path(__file__).resolve().parent.parent
RAW_TRANSCRIPT_FILE = BASE_DIR/"data"/"test_transcript.csv"
NO_BC_FILE = BASE_DIR/"results"/"no_backchannel_transcript.csv"
CLEANED_FILE = BASE_DIR/"results"/"cleaned_transcript.csv"
LABELED_CLEANED_FILE = BASE_DIR/"results"/"labeled_cleaned_transcript.csv"
OUTPUT_LABELED = BASE_DIR/"results"/"labeled_transcript.csv"

# LLM settings
api_url = "http://localhost:11434/api/generate"
ollama_model = "mistral:7b-instruct"
DEFAULT_MAX_TOK = 16
DEFAULT_TEMP = 0.0

context = """
The following utterances are excerpted from a conversation between two speakers who are speaking online via Zoom during the COVID-19 pandemic.
"""

def create_labeling_prompt(prev_utt, cur_utt, context) -> str:
    return f"""
You are a strict classifier for conversational utterances. You must assign one integer label from 1 to 5 based on the amount of cognitive and social effort required to make the utterance.

**LABELING SCALE:**
1 – The utterance requires little or  no cognitive effort, usually applies to very common lines people automatically say in conversations without thinking
5 – The utterance shows that the speaker is highly engaged in the conversation. The utterance could be personalized, often opinionated or providing an explanation, context-specific content, showing high cognitive and social effort. 

**RULES:**
- Output **exactly one** integer: **0**, **1**, **2**, **3**, **4**, or **5**.  
- Do **NOT** add any other text, punctuation, or justification.
- Do not be reserved in labelling utterances as **1** or **5**. Label as such if you think the utterance is very low or very high effort.

**Context:**  {context}

Now, classify the CURRENT utterance based on the previous utterance and the current utterance:
Previous utterance: "{prev_utt}"  
Current utterance to classify: "{cur_utt}"
"""

def call_ollama(prompt_1) -> str:
    """
    Send the prompt to the local LLM via Ollama and extract the label.

    The response is expected to contain a single digit (0–5). We allow the
    model to return extra text and extract only the first digit appearing in
    the response. If no digit is found, return "0".
    """

    payload = {
        "model": ollama_model,
        "prompt": prompt_1,
        "max_tokens": DEFAULT_MAX_TOK,
        "temperature": DEFAULT_TEMP,
        "stream": False
    }
    try:
        r = requests.post(api_url, json=payload, timeout=45)
        r.raise_for_status()
        resp = r.json().get("response", "").strip()
        # Extract the first digit from the response
        first_digit = None
        for ch in resp:
            if ch.isdigit():
                first_digit = ch
                break
        if first_digit and first_digit in {"0", "1", "2", "3", "4", "5"}:
            return first_digit
    except Exception:
        # If any error occurs, fall back to label "0"
        return "0"
    return "0"

def main():
    # load tables
    raw_df = pd.read_csv(RAW_TRANSCRIPT_FILE)
    no_bc_df = pd.read_csv(NO_BC_FILE)
    clean_df = pd.read_csv(CLEANED_FILE)

    raw_df["final_label"] = ""
    kept_ids = set(no_bc_df.turn_id)
    raw_df.loc[~raw_df.turn_id.isin(kept_ids), "final_label"] = "backchannel"

    # classify each merged utterance in cleaned_df
    clean_df = clean_df.reset_index(drop=True)
    clean_df["prev_utt"] = clean_df.utterance.shift(1).fillna("")
    clean_df["auto_label"] = ""
    for i, row in tqdm(clean_df.iterrows(), total=len(clean_df), desc="Labeling..."):
        label = call_ollama(create_labeling_prompt(row.prev_utt, row.utterance, context))
        clean_df.at[i, "auto_label"] = label

    clean_df.to_csv(LABELED_CLEANED_FILE, index=False)
    print(f"Wrote intermediate cleaned transcript with labels to {LABELED_CLEANED_FILE} with {len(clean_df)} rows")

    # spread labels back to every original utterance via source_rows
    for _, row in clean_df.iterrows():
        try:
            # use ast to evaluate the source_rows strings
            src = ast.literal_eval(row.source_rows)
        except Exception:
            continue
        for tid in src:
            raw_df.loc[raw_df.turn_id == tid, "final_label"] = row.auto_label

    # Replace any "unclassified" labels with "0"
    raw_df.loc[raw_df.final_label == "unclassified", "final_label"] = "0"
    # fallback for empty labels to be "0"
    raw_df.loc[raw_df.final_label == "", "final_label"] = "0"

    raw_df.to_csv(OUTPUT_LABELED, index=False)
    print(f"Wrote labelled transcript {OUTPUT_LABELED} with ({len(raw_df)} rows)")

if __name__ == "__main__":
    main()
