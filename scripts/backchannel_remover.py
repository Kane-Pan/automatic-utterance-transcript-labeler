import os
from pathlib import Path
import re, requests
import pandas as pd

BASE_DIR    = Path(__file__).resolve().parent.parent
INPUT_FILE  = BASE_DIR/"data"/"test_transcript.csv"
OUTPUT_FILE = BASE_DIR/"results"/"no_backchannel_transcript.csv"

API_ENDPOINT    = "http://localhost:11434/api/generate"
LLM_MODEL       = "mistral:7b-instruct"
DEFAULT_MAX_TOK = 40
DEFAULT_TEMP    = 0

filler_words = {
    "yeah", "yea", "yep", "uh-huh", "uh-uh", "uh", "mhm", "mm-hmm", "hmm",
    "oh", "wow", "right", "okay", "ok", "huh", "all", "all right"
}
answer_words = {"yes", "no", "yeah", "yep", "right", "ok", "okay", "sure", "uh-huh", "mhm", "mm-hmm"}

WORD_RE = re.compile(r"\w+\'?\w*")

def tokens(txt: str) -> list[str]:
    return WORD_RE.findall(txt.lower())

def whole_line_is_filler(row: pd.Series) -> bool:
    return set(tokens(row.utterance)).issubset(filler_words)

def is_noise_word(row: pd.Series, prev: pd.Series | None) -> bool:
    if row.n_words != 1:
        return False
    toks = tokens(row.utterance)
    word = toks[0] if toks else ""
    if word in filler_words | answer_words:
        return False
    if prev is not None and (prev.end_question or prev.questions):
        return False
    return True

def build_bc_prompt(prev_utt: str, cur_utt: str, next_utt: str) -> str:
    return f"""Decide if the utterance next to CURRENT is a filler or back‑channel.
Remove it if it adds zero informational content.
Examples:
PREVIOUS: "Are you doing alright?"
CURRENT: "yeah"
KEEP (even though "yeah" is a filler, it answers the question so keep it)

PREVIOUS: "I see you have a new car."
CURRENT: "uh‑huh"
KEEP (even though "uh‑huh" is a filler, it afirms the statement like an answer to a question)

PREVIOUS: "I don't like the weather here."
CURRENT: "uh"
REMOVE ("uh" is a filler and it adds no content and is not an answer)

Reply with just KEEP or REMOVE.

PREVIOUS: "{prev_utt}"
CURRENT: "{cur_utt}"
NEXT: "{next_utt}"
"""

def ask_llm_for_keep(prev_txt, cur_txt, next_txt) -> str:
    prompt = build_bc_prompt(prev_txt, cur_txt, next_txt)
    try:
        r = requests.post(
            API_ENDPOINT,
            json={
                "model": LLM_MODEL,
                "prompt": prompt,
                "temperature": DEFAULT_TEMP,
                "max_tokens": DEFAULT_MAX_TOK,
                "stream": False
            },
            timeout=45
        )
        r.raise_for_status()
        text = r.json().get("response", "")

        # sometimes returns extra fluff, so we take just the first word
        verdict = text.strip().split()[0].upper()
        if verdict in {"KEEP", "REMOVE"}:
            return verdict
    except Exception:
        return "KEEP" # default fallback
    
    return "KEEP"

def main():
    df = pd.read_csv(INPUT_FILE)
    keep_mask = []

    total = len(df)
    for i in range(total):
        row = df.iloc[i]

        # prev or next rows with explicit ifs
        prev = None
        if i > 0:
            prev = df.iloc[i - 1]

        next_row = None
        if i + 1 < total:
            next_row = df.iloc[i + 1]

        # decide keep/remove
        # True if keep, False if remove
        decision = None 

        if row.n_words >= 3 or row.end_question or row.questions:
            decision = True
        else:
            if whole_line_is_filler(row):
                decision = False
            else:
                if is_noise_word(row, prev):
                    decision = False
                else:
                    if row.n_words == 2:
                        decision = True
                    else:
                        # fallback to LLM
                        if prev is not None:
                            prev_txt = prev.utterance
                        else:
                            prev_txt = "<START>"

                        cur_txt = row.utterance

                        if next_row is not None:
                            next_txt = next_row.utterance
                        else:
                            next_txt = "<END>"

                        verdict = ask_llm_for_keep(prev_txt, cur_txt, next_txt)
                        if verdict == "KEEP":
                            decision = True
                        else:
                            decision = False

        keep_mask.append(decision)

    cleaned = df.loc[keep_mask].copy()
    cleaned.to_csv(OUTPUT_FILE, index=False)
    print(f"wrote no‑backchannel transcript {OUTPUT_FILE} with ({len(cleaned)} rows)")


if __name__ == "__main__":
    main()