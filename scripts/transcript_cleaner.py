import os
from pathlib import Path
import re, requests
import pandas as pd
from tqdm import tqdm  

# Step 1: remove backchannels from the transcript

BASE_DIR = Path(__file__).resolve().parent.parent
INPUT_FILE = BASE_DIR/"data"/"test_transcript.csv"
OUTPUT_FILE = BASE_DIR/"results"/"no_backchannel_transcript.csv"

API_ENDPOINT = "http://localhost:11434/api/generate"
LLM_MODEL = "mistral:7b-instruct"
DEFAULT_MAX_TOK = 40
DEFAULT_TEMP = 0

filler_words = {
    "yeah", "yea", "yep", "uh-huh", "uh-uh", "uh", "mhm", "mm-hmm", "hmm",
    "oh", "wow", "right", "okay", "ok", "huh", "all", "all right"
}
answer_words = {"yes", "no", "yeah", "yep", "right", "ok", "okay", "sure", "uh-huh", "mhm", "mm-hmm"}

WORD_RE = re.compile(r"\w+\'?\w*")

def tokens(text) -> list[str]:
    return WORD_RE.findall(text.lower())

def whole_line_is_filler(row: pd.Series):
    return set(tokens(row.utterance)).issubset(filler_words)

def is_noise_word(row, prev):
    if row.n_words != 1:
        return False
    toks = tokens(row.utterance)
    word = toks[0] if toks else ""
    if word in filler_words or word in answer_words:
        return False
    if prev is not None and (prev.end_question or prev.questions):
        return False
    return True

def build_bc_prompt(prev_utt, cur_utt, next_utt):
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

def run_backchannel():
    df = pd.read_csv(INPUT_FILE)
    keep_mask = []

    total = len(df)
    for i in range(total):
        row = df.iloc[i]

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
    print(f"Wrote no‑backchannel transcript {OUTPUT_FILE} with ({len(cleaned)} rows)")

# Step 2: merge utterances if necessary

# file paths
BASE_DIR_MERGE = Path(__file__).resolve().parent.parent  
INPUT_FILE_MERGE = BASE_DIR_MERGE/"results"/"no_backchannel_transcript.csv"  
OUTPUT_FILE_MERGE = BASE_DIR_MERGE/"results"/"cleaned_transcript.csv"  

# LLM settings
API_ENDPOINT_MERGE = "http://localhost:11434/api/generate"  
LLM_MODEL_MERGE = "mistral:7b-instruct"  
DEFAULT_MAX_TOK_MERGE = 30  
DEFAULT_TEMP_MERGE = 0

SHORT_SKIP_DUR = 2.0  
SKIP_WORDS_LIMIT = 3  
MERGE_PAUSE = 0.6  
MEDIUM_GAP = 3.0  

# Punctuation sets  
PRIMARY_PUNCTS   = {".", "?", "!"}  
SECONDARY_PUNCTS = {'"', "”", "'", ")"}  

def has_sentence_end(text):  
    # Strip trailing punctuation and check if it ends with a primary punctuation
    t = text.rstrip()  
    while t and t[-1] in SECONDARY_PUNCTS:  
        t = t[:-1]  
    return bool(t) and (t[-1] in PRIMARY_PUNCTS)  

def build_merge_prompt(prev_utt, cur_utt):   
    return f"""The following are two successive utterances by the same speaker.
If the PREVIOUS sentence seems to be unfinished and/or the CURRENT idea completes the PREVIOUS sentence or idea, reply MERGE.
Otherwise reply KEEP.

Examples:
PREVIOUS: "I went to the store, and I bought some milk"
CURRENT: "and uh some bread."
MERGE (the previous sentence is finished, but the current one is an obvious continuation that adds to the previous idea)

PREVIOUS: "I went to the store, and I bought some milk."
CURRENT: "I also bought some bread."
KEEP (the previous sentence is finished and the current one is a new sentence that does not obviously continue the previous sentence)

PREVIOUS: "I bought a pair of shoes"
CURRENT: ",a week ago."
MERGE (the previous sentence is finished, but the current one is an obvious continuation that adds new information to the previous idea)

Your turn: reply with just MERGE or KEEP.
PREVIOUS: "{prev_utt}"
CURRENT: "{cur_utt}"
"""
   

def ask_llm_for_merge(prev_text, cur_text):  
     
    prompt = build_merge_prompt(prev_text, cur_text)  
    try:  
        r = requests.post(API_ENDPOINT_MERGE, json={  
            "model": LLM_MODEL_MERGE,  
            "prompt": prompt,  
            "temperature": DEFAULT_TEMP_MERGE,  
            "max_tokens": DEFAULT_MAX_TOK_MERGE,  
            "stream": False,  
        }, timeout=50) 

        r.raise_for_status()  
        body = r.json().get("response", "")

        # sometimes returns extra fluff, so we take just the first word 
        decision = body.strip().split()[0].upper()  
        if decision in ("MERGE", "KEEP"):  
            return decision  
    except Exception:  
        return "KEEP"
    
    return "KEEP"  

def is_incomplete_segment(txt):  
    # Double-check if utterance ends cleanly  
    return (not has_sentence_end(txt))

def can_skip_interjection(row, base_unfinished):    
    duration = row.stop - row.start  
    if duration > SHORT_SKIP_DUR or row.n_words > SKIP_WORDS_LIMIT:  
        return False  
    if base_unfinished:  
        return True  
    
    # only keep if question-like  
    return not (row.end_question or row.questions)  

def decides_merge(r1, r2):  
    pause = r2.start - r1.stop  
    # immediate continuation or dangling sentence  
    if pause < MERGE_PAUSE or is_incomplete_segment(r1.utterance):  
        return True  
    # medium gap/not obvious -> ask LLM  
    if pause < MEDIUM_GAP:  
        return ask_llm_for_merge(r1.utterance, r2.utterance) == "MERGE"  

    return False  

def run_merger():  
 
    transcript_df = pd.read_csv(INPUT_FILE_MERGE)  
    output_list = []  
    idx = 0  
    pbar = tqdm(total=len(transcript_df), desc="Merging")  

    while idx < len(transcript_df):  
        curr = transcript_df.iloc[idx].copy()  
        curr["source_rows"] = [int(curr.turn_id)]  
        idx2 = idx + 1  

        # attempt to merge next rows  
        while idx2 < len(transcript_df):  
            nxt = transcript_df.iloc[idx2]  

            if nxt.speaker == curr.speaker:  
                if decides_merge(curr, nxt):
                    # merge the utterances and update the current row
                    curr.utterance += " " + nxt.utterance.lstrip()  
                    curr.stop = nxt.stop  
                    curr.n_words += nxt.n_words  
                    curr.source_rows.append(int(nxt.turn_id))  
                    idx2 += 1  
                    continue  
                # Note: could tweak threshold here if needed  
                break  

            # skip small interjections  
            if can_skip_interjection(nxt, is_incomplete_segment(curr.utterance)):  
                if idx2 + 1 < len(transcript_df) and transcript_df.iloc[idx2+1].speaker == curr.speaker:  
                    maybe = transcript_df.iloc[idx2+1]  
                    if decides_merge(curr, maybe):  
                        # skipping the interjection  
                        idx2 += 1  
                        continue  
            break  

        output_list.append(curr)  
        pbar.update(idx2 - idx)  
        idx = idx2  

    pbar.close()   
    result_df = pd.DataFrame(output_list)  
    result_df.to_csv(OUTPUT_FILE_MERGE, index=False)  
    print(f"Merged transcript at {OUTPUT_FILE_MERGE} with {len(result_df)} rows")  

# run the scripts
if __name__ == "__main__":
    run_backchannel()
    run_merger()
