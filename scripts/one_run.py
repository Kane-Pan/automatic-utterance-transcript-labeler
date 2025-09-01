import os
from pathlib import Path
import re
import pandas as pd
from tqdm import tqdm

# Step 1: remove backchannels from the transcript

WORD_RE = re.compile(r"\w+\'?\w*")

def tokens(text) -> list[str]:
    return WORD_RE.findall(text.lower())

def whole_line_is_filler(row: pd.Series):
    return set(tokens(row.utterance)).issubset(filler_words)

def is_noise_word(row, prev):
    """
    One-word utterances are ONLY removed if they are pure backchannel/filler
    and are not direct answers to a question (based on metadata).
    """
    if row.n_words != 1:
        return False
    toks = tokens(row.utterance)
    word = toks[0] if toks else ""

    # Keep single-word answers if the previous turn was a question
    if prev is not None and (prev.end_question or prev.questions):
        return False

    # Remove if it's a filler/backchannel
    if word in filler_words:
        return True
    else:
        return False

def run_backchannel():
    df = pd.read_csv(INPUT_FILE)
    keep_decisions = []

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
                # Single-word: keep unless it's a filler/backchannel (not answering a question)
                if row.n_words == 1:
                    decision = not is_noise_word(row, prev)
                # Two-word: remove only if BOTH words are fillers AND not answering a question
                elif row.n_words == 2:
                    toks = set(tokens(row.utterance))
                    if prev is not None and (prev.end_question or prev.questions):
                        decision = True
                    else:
                        decision = not (toks and toks.issubset(filler_words))
                else:
                    # Fallback: keep
                    decision = True

        keep_decisions.append(decision)

    cleaned = df.loc[keep_decisions].copy()
    cleaned.to_csv(OUTPUT_FILE, index=False)
    print(f"Wrote no-backchannel transcript {OUTPUT_FILE} with ({len(cleaned)} rows)")

# Step 2: merge utterances if necessary

# Merge thresholds

# Punctuation sets

def has_sentence_end(text):
    # Strip trailing punctuation and check if it ends with a primary punctuation
    t = text.rstrip()
    while t and t[-1] in SECONDARY_PUNCTS:
        t = t[:-1]
    return bool(t) and (t[-1] in PRIMARY_PUNCTS)

def is_incomplete_segment(txt):
    # check if utterance ends cleanly
    return (not has_sentence_end(txt))

# Simple possible connectors that may indicate continuation

# common discourse markers to avoid triggering merges

def starts_with_connector(text: str) -> bool:
    t = text.lstrip()
    if not t:
        return False
    toks = tokens(t)
    return bool(toks) and toks[0] in CONNECTORS

# identify discourse-marker starts
def starts_with_discourse_marker(text: str) -> bool:
    t = text.lstrip()
    if not t:
        return False
    toks = tokens(t)
    return bool(toks) and toks[0] in DISCOURSE_MARKERS


def can_skip_interjection(row, base_unfinished):
    duration = row.stop - row.start
    if duration > SHORT_SKIP_DUR or row.n_words > SKIP_WORDS_LIMIT:
        return False

    toks = set(tokens(row.utterance))
    is_filler = toks.issubset(filler_words)
    has_overlap = bool(getattr(row, "overlap", False))

    if base_unfinished and (is_filler or has_overlap):
        return True

    return is_filler

def decides_merge(r1, r2):
    """
    Determine whether two utterances by the same speaker should be merged,
    using timing, length, punctuation, and simple cues.
    """
    pause = r2.start - r1.stop
    prev_incomplete = is_incomplete_segment(r1.utterance)
    cur_connector = starts_with_connector(r2.utterance)
    cur_discourse = starts_with_discourse_marker(r2.utterance)

    # Immediate continuation or dangling sentence
    if pause < MERGE_PAUSE:
        return True
    if prev_incomplete:
        return True

    # Avoid merging into a cleanly-ended utterance followed by a fresh question
    if getattr(r2, "questions", False) and has_sentence_end(r1.utterance):
        if not (pause < 0.5 and cur_connector):
            return False

    # Medium gap heuristic: syntactic connector or short-short continuation
    if pause < MEDIUM_GAP:
        if cur_connector:
            return True
        if (r1.n_words <= 4 and r2.n_words <= 6):
            return True
        if cur_discourse:
            return False

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
                break

            # handle small interjections from the other speaker
            if can_skip_interjection(nxt, is_incomplete_segment(curr.utterance)):
                # Determine if the interjection is a simple filler/back-channel based on tokens and length
                interjection_tokens = tokens(nxt.utterance)

                # Do not skip meaningful utterances solely based on length.
                is_filler_interjection = set(interjection_tokens).issubset(filler_words)
                if is_filler_interjection:
                    # consider merging across it
                    if idx2 + 1 < len(transcript_df):
                        maybe = transcript_df.iloc[idx2 + 1]
                        if decides_merge(curr, maybe):
                            # Skip the interjection and merge the next utterance
                            idx2 += 1
                            continue
                break

            break

        output_list.append(curr)
        pbar.update(idx2 - idx)
        idx = idx2

    pbar.close()
    result_df = pd.DataFrame(output_list)
    result_df.to_csv(OUTPUT_FILE_MERGE, index=False)
    print(f"Merged transcript at {OUTPUT_FILE_MERGE} with {len(result_df)} rows")

# Step 3: label utterances using LLM

import requests
import ast
from prompts import prompt_list

# LLM settings

context = """
The following utterances are excerpted from a conversation between two speakers who are speaking online via Zoom during the COVID-19 pandemic.
"""

def create_labeling_prompt(prev_utt, cur_utt, prompt_num) -> str:
    prompt_template = prompt_list[prompt_num]
    return prompt_template.format(prev_utt=prev_utt, cur_utt=cur_utt)

def call_ollama(prompt) -> str:
    """
    Send the prompt to the local LLM via Ollama and extract the label.

    The response is expected to contain a single digit (0–5). We allow the
    model to return extra text and extract only the first digit appearing in
    the response. If no digit is found, return "0".
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

def run_labeler():
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
        label = call_ollama(create_labeling_prompt(row.prev_utt, row.utterance, prompt_id))
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

# Step 4: get stats

import matplotlib.pyplot as plt
from sklearn.metrics import cohen_kappa_score, confusion_matrix, ConfusionMatrixDisplay
from scipy.stats import kendalltau

def run_stats():
    auto_df = pd.read_csv(AUTO_FILE)
    manual_df = pd.read_csv(MANUAL_FILE)

    df = pd.merge(
        auto_df[['turn_id', 'final_label']].rename(columns={'final_label': 'auto_label'}),
        manual_df[['turn_id', 'final_label']].rename(columns={'final_label': 'manual_label'}),
        on='turn_id',
        how='inner'
    )


    for col in ['auto_label', 'manual_label']:
        df[col] = df[col].astype(str)  # make sure it's a string
        df[col] = df[col].str.lower()
        df[col] = df[col].replace('backchannel', '0') # replace backchannels with '0' for comparison

    auto = pd.to_numeric(df['auto_label'], errors='coerce')
    manual = pd.to_numeric(df['manual_label'], errors='coerce')

    # Filter out backchannels
    mask = (auto > 0) & (manual > 0)
    auto = auto[mask]
    manual = manual[mask]

    tau = kendalltau(auto, manual, nan_policy='omit')
    kappa = cohen_kappa_score(auto, manual, labels=[1, 2, 3, 4, 5], weights='quadratic')
    C = confusion_matrix(manual, auto, labels=[1, 2, 3, 4, 5])

    print(f"Kendall's Tau: {tau.statistic:.4f}, p-value: {tau.pvalue:.4g}")
    print(f"Cohen's Kappa: {kappa:.4f}")

    disp = ConfusionMatrixDisplay(confusion_matrix=C, display_labels= [1, 2, 3, 4, 5])
    disp.plot(values_format='d')
    plt.title("Manual vs AI Utterance Labels (Counts)")
    plt.xlabel("AI-predicted label")
    plt.ylabel("Manually assigned label")
    plt.tight_layout()
    plt.show()

    C_norm = confusion_matrix(manual, auto, labels=[1, 2, 3, 4, 5], normalize='true')
    disp_norm = ConfusionMatrixDisplay(confusion_matrix=C_norm, display_labels=[1, 2, 3, 4, 5])
    disp_norm.plot(values_format='.2f')
    plt.title("Manual vs AI Utterance Labels (Row-Normalized)")
    plt.xlabel("AI-predicted label")
    plt.ylabel("Manually assigned label")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    BASE_DIR = Path(__file__).resolve().parent.parent
    INPUT_FILE = BASE_DIR/"data"/"test_transcript.csv"
    OUTPUT_FILE = BASE_DIR/"results"/"no_backchannel_transcript.csv"

    filler_words = {
        "yeah", "yea", "yep", "uh-huh", "uh-uh", "uh", "mhm", "mm-hmm", "hmm",
        "oh", "wow", "right", "okay", "ok", "huh", "all", "all right"
    }
    answer_words = {"yes", "no", "yeah", "yep", "right", "ok", "okay", "sure", "uh-huh", "mhm", "mm-hmm"}

    BASE_DIR_MERGE = Path(__file__).resolve().parent.parent
    INPUT_FILE_MERGE = BASE_DIR_MERGE/"results"/"no_backchannel_transcript.csv"
    OUTPUT_FILE_MERGE = BASE_DIR_MERGE/"results"/"cleaned_transcript.csv"

    SHORT_SKIP_DUR = 0.7
    SKIP_WORDS_LIMIT = 2
    MERGE_PAUSE = 1.2
    MEDIUM_GAP = 2.5

    PRIMARY_PUNCTS = {".", "?", "!"}
    SECONDARY_PUNCTS = {'"', "”", "'", ")"}

    CONNECTORS = {
        "and", "but", "so", "because", "then", "also", "or", "nor", "plus",
        "except", "that", "which", "who", "where", "when", "while", "if",
        "though", "although"
    }
    DISCOURSE_MARKERS = {"yeah", "okay", "ok", "right", "well", "uh", "um"}

    RAW_TRANSCRIPT_FILE = BASE_DIR/"data"/"test_transcript.csv"
    NO_BC_FILE = BASE_DIR/"results"/"no_backchannel_transcript.csv"
    CLEANED_FILE = BASE_DIR/"results"/"cleaned_transcript.csv"
    LABELED_CLEANED_FILE = BASE_DIR/"results"/"labeled_cleaned_transcript.csv"
    OUTPUT_LABELED = BASE_DIR/"results"/"labeled_transcript.csv"

    api_url = "http://localhost:11434/api/generate"
    ollama_model = "hf.co/bartowski/Meta-Llama-3.1-8B-Instruct-GGUF:Q4_K_M"
    DEFAULT_MAX_TOK = 16
    DEFAULT_TEMP = 0.0
    prompt_id = "soc_1" # choose which prompt to use from prompts.py

    MANUAL_FILE = BASE_DIR/"data"/"ChatGPT_labeled_transcript.csv"
    AUTO_FILE = BASE_DIR/"results"/"labeled_transcript.csv"

    # Run each step
    run_backchannel()
    run_merger()
    run_labeler()
    run_stats()
