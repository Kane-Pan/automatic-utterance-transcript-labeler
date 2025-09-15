import os
from pathlib import Path
import pandas as pd
from tqdm import tqdm
import requests

#create prompt, separate for info-transfer and relationship-building scores
def create_labeling_prompt(prev_utt, cur_utt, label_type) -> str:
    """
    Build a prompt for one 1–5 label.
    label_type: "info" or "social"
    """
    if label_type == "info":
        prompt_text = (
            "You are a strict classifier for conversational utterances. "
            "You must assign one integer label from 1 to 5 based on the following criteria:\n"
            "How much new information does this utterance transfer?\n"
            "Please rate on a scale from 1 to 5, where 1 means no new information and 5 means a high amount of new information.\n\n"
            "1: None. The sentence does not convey any new facts or data. It might be a simple greeting, a filler word, or a purely social gesture.\n\n"
            "2: Minimal. The sentence conveys a very minor piece of information, such as a simple confirmation or a detail that isn't central to the conversation's topic.\n\n"
            "3: Moderate. The sentence provides some clear, new information that contributes to the conversation's content.\n\n"
            "4: Substantial. The sentence's primary purpose is to deliver a significant amount of new or important information.\n\n"
            "5: High. The sentence is almost exclusively about transferring critical, dense, or central information."
        )
        tail = "Based on the above definitions and examples, given the context from the previous utterance made by the other speaker, return only ONE digit 1–5 for the information-transfer score of the current utterance."
    else:
        prompt_text = (
            "You are a strict classifier for conversational utterances. "
            "You must assign one integer label from 1 to 5 based on the following criteria:\n"
            "How much does this sentence build or maintain a relationship between the speakers?\n"
            "Please rate on a scale from 1 to 5, where 1 means no social value and 5 means high social value.\n\n"
            "1: None. The sentence is purely transactional and has no social function.\n\n"
            "2: Minimal. The sentence has a very slight social function, such as a brief acknowledgment or a polite but perfunctory phrase.\n\n"
            "3: Moderate. The sentence helps to maintain the conversational flow and acknowledges the other person, showing moderate rapport.\n\n"
            "4: Substantial. The sentence is a clear effort to show empathy, build trust, or strengthen the connection between the speakers.\n\n"
            "5: High. The sentence's main purpose is to perform a significant social function, such as a heartfelt apology, a warm expression of thanks, or a profound emotional statement."
        )
        tail = "Based on the above definitions and examples, given the context from the previous utterance made by the other speaker, return only ONE digit 1–5 for the relationship-building score of the current utterance."

    prompt = (
        f"{prompt_text}\n\n"
        f"Previous utterance: {prev_utt}\n"
        f"Current utterance: {cur_utt}\n\n"
        f"{tail}"
    )
    return prompt


def call_ollama(prompt) -> str:
    """
    Send the prompt to the local LLM via Ollama and extract the label (1–5).
    If no valid digit is found or an error occurs, return '1'.
    """
    payload = {
        "model": ollama_model,
        "prompt": prompt,
        "max_tokens": DEFAULT_MAX_TOK,
        "temperature": DEFAULT_TEMP,
        "stream": False
    }
    try:
        r = requests.post(api_url, json=payload, timeout=45)
        r.raise_for_status()
        resp = r.json().get("response", "").strip()
        for ch in resp:
            if ch in {"1", "2", "3", "4", "5"}:
                return ch
    except Exception:
        return "1"
    return "1"

def run_labeler():
    # read the new CALLHOME-style csv: columns are "speaker" and "utterance"
    transcript = pd.read_csv(CALLHOME_FILE)
    transcript = transcript.reset_index(drop=True)

    # add output columns
    transcript["social_score"] = ""
    transcript["info_score"] = ""

    # go line-by-line, use previous utterance as context
    for i, row in tqdm(transcript.iterrows(), total=len(transcript), desc="Labeling..."):
        prev_utt = transcript.at[i-1, "utterance"] if i > 0 else ""

        info_prompt = create_labeling_prompt(prev_utt, row.utterance, "info")
        social_prompt = create_labeling_prompt(prev_utt, row.utterance, "social")

        info_label = call_ollama(info_prompt)
        social_label = call_ollama(social_prompt)

        transcript.at[i, "info_score"] = info_label
        transcript.at[i, "social_score"] = social_label

    transcript.to_csv(OUTPUT_LABELED, index=False)
    print(f"Wrote labelled transcript {OUTPUT_LABELED} with ({len(transcript)} rows)")

if __name__ == "__main__":
    BASE_DIR = Path(__file__).resolve().parent.parent

    # input/output for the CALLHOME-format csv
    CALLHOME_FILE = BASE_DIR/"data"/"CALLHOME_TEST.csv"
    OUTPUT_LABELED = BASE_DIR/"results"/"labeled_transcript.csv"

    api_url = "http://localhost:11434/api/generate"
    ollama_model = "hf.co/bartowski/Meta-Llama-3.1-8B-Instruct-GGUF:Q4_K_M"
    DEFAULT_MAX_TOK = 16
    DEFAULT_TEMP = 0.0

    run_labeler()
