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
OUTPUT_LABELED = BASE_DIR/"results"/"labeled_transcript.csv"

# LLM settings
api_url = "http://localhost:11434/api/generate"
ollama_model = "mistral:7b-instruct"
DEFAULT_MAX_TOK = 50
DEFAULT_TEMP = 0.0

def create_labeling_prompt(prev_utt, cur_utt) -> str:
    return f"""
You are a strict classifier for conversational utterances. Output ONLY one of two labels:

AUTOMATIC - if the utterance sounds scripted, routine, formulaic, or is a boilerplate social line.
             Examples: "Hi, how are you?", "Where are you from?", "Would you like a receipt?", 
             "Welcome to our store, how may I help you?", "Good morning", "I'm fine, thank you."

NON-AUTOMATIC - everything else. These lines show spontaneity, personalization, or context-specific content,
                even if they're short.

RULES:
1. Your very first token must be exactly AUTOMATIC or NON-AUTOMATIC (all caps, hyphen OK).
2. After the label, you may give a short justification (one sentence max). Do not add anything before the label.

Quick checklist you must use before labeling:
- Does it resemble a canned phrase a cashier, receptionist, or polite stranger would say?
- Is it a closed, routine question about mundane info without personalization?
If YES, label AUTOMATIC. Otherwise, label NON-AUTOMATIC.

Previous utterance from other speaker for context: "{prev_utt}"
CURRENT utterance (to be labeled by you): "{cur_utt}"
"""

def call_ollama(prompt_1) -> str:
    payload = {
        "model": ollama_model,
        "prompt": prompt_1,
        "max_tokens": DEFAULT_MAX_TOK,
        "temperature": DEFAULT_TEMP,
        "stream": False
    }
    r = requests.post(api_url, json=payload, timeout=45)
    r.raise_for_status()
    resp = r.json().get("response", "").strip().upper()
    if resp.startswith("AUTOMATIC"):
        return "automatic"
    if resp.startswith("NON-AUTOMATIC") or resp.startswith("NON"):
        return "non-automatic"
    # fallback
    return "non-automatic"

def main():
    # load all tables
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
        label = call_ollama(create_labeling_prompt(row.prev_utt, row.utterance))
        clean_df.at[i,"auto_label"] = label

    # spread labels back to every original utterance via source_rows
    for _, row in clean_df.iterrows():
        try:
            # use ast to evaluate the source_rows strings
            src = ast.literal_eval(row.source_rows)
        except Exception:
            continue
        for tid in src:
            raw_df.loc[raw_df.turn_id==tid, "final_label"] = row.auto_label

    # fallback for empty labeels to be "unclassified"
    raw_df.loc[raw_df.final_label=="", "final_label"] = "unclassified"

    raw_df.to_csv(OUTPUT_LABELED, index=False)
    print(f"Wrote labelled transcript {OUTPUT_LABELED} with ({len(raw_df)} rows)")

if __name__=="__main__":
    main()
