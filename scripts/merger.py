import os
from pathlib import Path
import re, requests
import pandas as pd
from tqdm import tqdm

# Single-step cleaner/merger for transcripts with only: speaker, utterance

BASE_DIR = Path(__file__).resolve().parent.parent
INPUT_FILE = BASE_DIR/"data"/"CALLHOME_TEST.csv"
OUTPUT_FILE_MERGE = BASE_DIR/"results"/"cleaned_transcript.csv"

API_MERGE = "http://localhost:11434/api/generate"
LLM_MODEL_MERGE = "hf.co/bartowski/Meta-Llama-3.1-8B-Instruct-GGUF:Q4_K_M"
DEFAULT_MAX_TOK_MERGE = 30
DEFAULT_TEMP_MERGE = 0

PRIMARY_PUNCTS = {".", "?", "!"}
SECONDARY_PUNCTS = {'"', "”", "'", ")"}

# simple conjunction list based on CALLHOME-style data
CONJUNCTION_STARTS = {"and", "but", "so", "then", "or", "because", "well", "anyway", "also"}

# short backchannels (<= 2 words) we may drop if interrupting a continuation
filler_words = {
    "yeah", "yea", "yep", "uh-huh", "uh-uh", "uh", "mhm", "mm-hmm", "hmm",
    "oh", "wow", "right", "okay", "ok", "huh", "all", "all right"
}

WORD_RE = re.compile(r"\w+\'?\w*")

def tokens(text):
    return WORD_RE.findall(str(text).lower())

def has_sentence_end(text):
    t = str(text).rstrip()
    while t and t[-1] in SECONDARY_PUNCTS:
        t = t[:-1]
    return bool(t) and (t[-1] in PRIMARY_PUNCTS)

def normalize_ws(s):
    return " ".join(str(s).strip().split())

def starts_with_conjunction(text):
    t = normalize_ws(text).lower().lstrip(", ")
    if not t:
        return False
    first = t.split()[0]
    return first in CONJUNCTION_STARTS

def is_cutoff_or_dangling(prev_text):
    s = str(prev_text).strip()
    if not s:
        return False
    if s.endswith("-") or s.endswith(","):
        return True
    tail = s.lower().rstrip(" .!?")
    return tail.endswith((" and"," but"," so"))

def seeks_response(prev_text):
    s = normalize_ws(prev_text).lower()
    if s.endswith("?"):
        return True
    # light patterns like “you’re kidding”, “really”
    if s.startswith(("are you","do you","did you","have you","is it","is he","is she",
                     "can you","could you","would you","should you","how's that","how’s that",
                     "you're kidding","youre kidding","really")):
        return True
    if s.endswith((" right?", " okay?")):
        return True
    return False

def boundary_repeat(prev_text, next_text, k=3):
    strip = lambda w: w.strip(".,!?").lower()
    prev_tokens = [strip(w) for w in str(prev_text).split()]
    next_tokens = [strip(w) for w in str(next_text).split()]
    for n in range(min(k, len(prev_tokens), len(next_tokens)), 0, -1):
        if prev_tokens[-n:] == next_tokens[:n]:
            return True
    return False

def interjection_is_short_backchannel(utt):
    # only consider deletion if <= 2 words and subset of filler set
    w = tokens(utt)
    if len(w) == 0 or len(w) > 2:
        return False
    return set(w).issubset(filler_words)

def build_merge_prompt(prev_utt, cur_utt, context_before="", context_after=""):
    return f"""The following are two successive utterances by the same speaker.
If the PREVIOUS sentence seems unfinished and/or CURRENT clearly continues the idea, reply MERGE else KEEP.

CONTEXT BEFORE: "{context_before}"
PREVIOUS: "{prev_utt}"
CURRENT: "{cur_utt}"
CONTEXT AFTER: "{context_after}"
"""

def ask_llm_for_merge(prev_text, cur_text, context_before="", context_after=""):
    prompt = build_merge_prompt(prev_text, cur_text, context_before, context_after)
    try:
        r = requests.post(API_MERGE, json={
            "model": LLM_MODEL_MERGE,
            "prompt": prompt,
            "temperature": DEFAULT_TEMP_MERGE,
            "max_tokens": DEFAULT_MAX_TOK_MERGE,
            "stream": False,
        }, timeout=50)
        r.raise_for_status()
        body = r.json().get("response", "")
        decision = body.strip().split()[0].upper()
        if decision in ("MERGE", "KEEP"):
            return decision
    except Exception:
        return "KEEP"
    return "KEEP"

def is_incomplete_segment(txt):
    # unfinished if no clean end or dangling/cutoff
    return (not has_sentence_end(txt)) or is_cutoff_or_dangling(txt)

def decides_merge(prev_row, next_row, context_before="", context_after=""):

    prev_u = prev_row.utterance
    next_u = next_row.utterance

    # do NOT merge if prev seeks a response (e.g., '?', “you're kidding”)
    if seeks_response(prev_u):
        return False
    # merge if next starts with a conjunction
    if starts_with_conjunction(next_u):
        return True
    # merge if previous looks unfinished
    if is_incomplete_segment(prev_u):
        return True
    # merge if boundary has small token repeat
    if boundary_repeat(prev_u, next_u):
        return True

    # LLM fallback
    return ask_llm_for_merge(prev_u, next_u, context_before, context_after) == "MERGE"

def run_merger():
    df = pd.read_csv(INPUT_FILE)
    output_list = []
    idx = 0
    pbar = tqdm(total=len(df), desc="Merging")

    while idx < len(df):
        curr = df.iloc[idx].copy()
        # keep a simple trace of source indices (no turn_id in this format)
        curr["source_rows"] = [int(idx)]
        idx2 = idx + 1

        while idx2 < len(df):
            nxt = df.iloc[idx2]

            # same speaker: consider merge
            if nxt.speaker == curr.speaker:
                ctx_before = "<START>" if idx == 0 else df.iloc[idx - 1].utterance
                ctx_after = "<END>" if (idx2 + 1) >= len(df) else df.iloc[idx2 + 1].utterance
                if decides_merge(curr, nxt, ctx_before, ctx_after):
                    curr.utterance += " " + str(nxt.utterance).lstrip()
                    curr.source_rows.append(int(idx2))
                    idx2 += 1
                    continue
                break

            # different speaker: optionally drop a short interjection if A–B–A continuation
            if interjection_is_short_backchannel(nxt.utterance) and not seeks_response(curr.utterance):
                if idx2 + 1 < len(df) and df.iloc[idx2 + 1].speaker == curr.speaker:
                    maybe = df.iloc[idx2 + 1]
                    # allow skip if base unfinished or next clearly continues
                    if is_incomplete_segment(curr.utterance) or starts_with_conjunction(maybe.utterance) or boundary_repeat(curr.utterance, maybe.utterance):
                        curr.utterance += " " + str(maybe.utterance).lstrip()
                        curr.source_rows.append(int(idx2 + 1))
                        idx2 += 2
                        continue
                break  # can't skip → emit curr

            break  # different speaker and not skip-worthy → emit curr

        output_list.append(curr)
        pbar.update(idx2 - idx)
        idx = idx2

    pbar.close()
    result_df = pd.DataFrame(output_list)
    result_df[["speaker", "utterance"]].to_csv(OUTPUT_FILE_MERGE, index=False)
    print("Merged transcript at", OUTPUT_FILE_MERGE, "with", len(result_df), "rows")

if __name__ == "__main__":
    run_merger()
