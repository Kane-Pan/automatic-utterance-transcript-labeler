import os
from pathlib import Path
import re, requests
import pandas as pd
from tqdm import tqdm

# Step 1: remove backchannels from the transcript

BASE_DIR = Path(__file__).resolve().parent.parent
INPUT_FILE = BASE_DIR/"data"/"CALLHOME_TEST.csv"
OUTPUT_FILE = BASE_DIR/"results"/"no_backchannel_transcript.csv"

API = "http://localhost:11434/api/generate"
LLM_MODEL = "mistral:7b-instruct"
DEFAULT_MAX_TOK = 40
DEFAULT_TEMP = 0

filler_words = {
    "yeah", "yea", "yep", "uh-huh", "uh-uh", "uh", "mhm", "mm-hmm", "hmm",
    "oh", "wow", "right", "okay", "ok", "huh", "all", "all right", "yup", "nope"
}
answer_words = {"yes", "no", "yeah", "yep", "right", "ok", "okay", "sure", "uh-huh", "mhm", "mm-hmm", "yup", "nope"}

WORD_RE = re.compile(r"\w+\'?\w*")
TAG_RE = re.compile(r"\[[^\]]+\]")  # matches [UNINTELLIGIBLE], [DISTORTION], etc.

def tokens(text) -> list[str]:
    return WORD_RE.findall(str(text).lower())

def strip_tags(text: str) -> str:
    return TAG_RE.sub(" ", str(text))

def ends_with_q(text: str) -> bool:
    t = str(text).strip()
    return t.endswith("?")

def count_questions(text: str) -> int:
    return str(text).count("?")

def whole_line_is_filler(row: pd.Series):
    return set(tokens(strip_tags(row.utterance))).issubset(filler_words)

def tag_only_noise(text: str) -> bool:
    core = strip_tags(text)
    core_toks = tokens(core)
    if len(core_toks) == 0:
        return True
    if len(core_toks) == 1 and core_toks[0] in {"t", "uh", "um"}:
        return True
    return False

def is_noise_word(row, prev):
    if row.n_words != 1:
        return False
    toks = tokens(strip_tags(row.utterance))
    word = toks[0] if toks else ""
    if word in filler_words or word in answer_words:
        return False
    if prev is not None and (prev.end_question or prev.questions):
        return False
    return True

def build_bc_prompt(prev_utt, cur_utt, next_utt, context_before: str = "", context_after: str = ""):
    return f"""Decide if the utterance next to CURRENT is a filler or back-channel.
Remove it if it adds zero informational content.
Examples:
PREVIOUS: "Are you doing alright?"
CURRENT: "yeah"
KEEP (even though "yeah" is a filler, it answers the question so keep it)

PREVIOUS: "I see you have a new car."
CURRENT: "uh-huh"
KEEP (even though "uh-huh" is a filler, it affirms the statement like an answer to a question)

PREVIOUS: "I don't like the weather here."
CURRENT: "uh"
REMOVE ("uh" is a filler and it adds no content and is not an answer)

Reply with just KEEP or REMOVE.

EXTRA CONTEXT BEFORE: "{context_before}"
PREVIOUS: "{prev_utt}"
CURRENT: "{cur_utt}"
NEXT: "{next_utt}"
EXTRA CONTEXT AFTER: "{context_after}"
"""

def ask_llm_for_keep(prev_txt: str, cur_txt: str, next_txt: str,
                     context_before: str = "", context_after: str = "") -> str:
    prompt = build_bc_prompt(prev_txt, cur_txt, next_txt, context_before, context_after)
    try:
        r = requests.post(
            API,
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
        verdict = text.strip().split()[0].upper()
        if verdict in {"KEEP", "REMOVE"}:
            return verdict
    except Exception:
        return "KEEP"
    return "KEEP"

def run_backchannel():
    df = pd.read_csv(INPUT_FILE)
    df["utterance"] = df["utterance"].fillna("").astype(str)
    df["orig_line"] = df.index + 1
    df["n_words"] = df["utterance"].apply(lambda t: len(tokens(strip_tags(t))))
    df["end_question"] = df["utterance"].apply(ends_with_q)
    df["questions"] = df["utterance"].apply(count_questions)

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

        decision = None

        if tag_only_noise(row.utterance):
            decision = False
        elif row.n_words >= 3 or row.end_question or row.questions:
            decision = True
        else:
            if whole_line_is_filler(row):
                decision = False
            else:
                if is_noise_word(row, prev):
                    # Remove one-word utterances that are not answers or fillers
                    decision = False
                else:
                    if row.n_words == 2:
                        decision = True
                    else:
                        if prev is not None:
                            prev_txt = prev.utterance
                        else:
                            prev_txt = "<START>"

                        cur_txt = row.utterance

                        if next_row is not None:
                            next_txt = next_row.utterance
                        else:
                            next_txt = "<END>"

                        context_before = "<START>"
                        if i > 1:
                            context_before = df.iloc[i - 2].utterance
                        context_after = "<END>"
                        if i + 2 < total:
                            context_after = df.iloc[i + 2].utterance
                        verdict = ask_llm_for_keep(prev_txt, cur_txt, next_txt,
                                                    context_before, context_after)
                        if verdict == "KEEP":
                            decision = True
                        else:
                            decision = False

        keep_mask.append(decision)

    cleaned = df.loc[keep_mask, ["speaker", "utterance", "orig_line"]].reset_index(drop=True)
    cleaned.to_csv(OUTPUT_FILE, index=False)
    print(f"Wrote no-backchannel transcript {OUTPUT_FILE} with ({len(cleaned)} rows)")

# Step 2: merge utterances if necessary

# file paths
BASE_DIR_MERGE = Path(__file__).resolve().parent.parent
INPUT_FILE_MERGE = BASE_DIR_MERGE/"results"/"no_backchannel_transcript.csv"
OUTPUT_FILE_MERGE = BASE_DIR_MERGE/"results"/"cleaned_transcript.csv"

# LLM settings
API_MERGE = "http://localhost:11434/api/generate"
LLM_MODEL_MERGE = "mistral:7b-instruct"
DEFAULT_MAX_TOK_MERGE = 30
DEFAULT_TEMP_MERGE = 0

SKIP_WORDS_LIMIT = 5

# Punctuation sets
PRIMARY_PUNCTS = {".", "?", "!"}
SECONDARY_PUNCTS = {'"', "”", "'", ")"}

def has_sentence_end(text):
    # Strip trailing punctuation and check if it ends with a primary punctuation
    t = str(text).rstrip()
    while t and t[-1] in SECONDARY_PUNCTS:
        t = t[:-1]
    return bool(t) and (t[-1] in PRIMARY_PUNCTS)

def build_merge_prompt(prev_utt: str, cur_utt: str,
                       context_before: str = "", context_after: str = ""):
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

CONTEXT BEFORE: "{context_before}"
PREVIOUS: "{prev_utt}"
CURRENT: "{cur_utt}"
CONTEXT AFTER: "{context_after}"
"""

def ask_llm_for_merge(prev_text: str, cur_text: str,
                      context_before: str = "", context_after: str = ""):
    """
    Ask the local LLM whether to merge two successive utterances by the same speaker.
    """
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
    # check if utterance ends cleanly
    return (not has_sentence_end(txt))

def can_skip_interjection(text, base_unfinished):
    n_w = len(tokens(strip_tags(text)))
    if n_w > SKIP_WORDS_LIMIT:
        return False
    if base_unfinished:
        return True
    return set(tokens(strip_tags(text))).issubset(filler_words)

def decides_merge(prev_text: str, cur_text: str, context_before: str = "", context_after: str = ""):
    """
    Determine whether two utterances by the same speaker should be merged.
    """
    if is_incomplete_segment(prev_text):
        return True
    return ask_llm_for_merge(prev_text, cur_text, context_before, context_after) == "MERGE"

def run_merger():
    transcript_df = pd.read_csv(INPUT_FILE_MERGE)
    transcript_df["utterance"] = transcript_df["utterance"].fillna("").astype(str)

    output_rows = []
    idx = 0
    pbar = tqdm(total=len(transcript_df), desc="Merging")

    while idx < len(transcript_df):
        curr_speaker = transcript_df.iloc[idx].speaker
        curr_text = transcript_df.iloc[idx].utterance
        curr_lines = [int(transcript_df.iloc[idx].orig_line)] if "orig_line" in transcript_df.columns else [idx + 1]
        idx2 = idx + 1

        # attempt to merge next rows
        while idx2 < len(transcript_df):
            nxt_row = transcript_df.iloc[idx2]
            if nxt_row.speaker == curr_speaker:
                context_before_merge = "<START>" if idx == 0 else transcript_df.iloc[idx - 1].utterance
                context_after_merge = "<END>" if (idx2 + 1) >= len(transcript_df) else transcript_df.iloc[idx2 + 1].utterance
                if decides_merge(curr_text, nxt_row.utterance, context_before_merge, context_after_merge):
                    curr_text = (curr_text + " " + str(nxt_row.utterance).lstrip()).strip()
                    curr_lines.append(int(nxt_row.orig_line) if "orig_line" in transcript_df.columns else (idx2 + 1))
                    idx2 += 1
                    continue
                break

            if can_skip_interjection(nxt_row.utterance, is_incomplete_segment(curr_text)):
                if idx2 + 1 < len(transcript_df) and transcript_df.iloc[idx2 + 1].speaker == curr_speaker:
                    maybe_next = transcript_df.iloc[idx2 + 1]
                    context_before_merge = "<START>" if idx == 0 else transcript_df.iloc[idx - 1].utterance
                    context_after_merge = "<END>" if (idx2 + 2) >= len(transcript_df) else transcript_df.iloc[idx2 + 2].utterance
                    if decides_merge(curr_text, maybe_next.utterance, context_before_merge, context_after_merge):
                        curr_text = (curr_text + " " + str(maybe_next.utterance).lstrip()).strip()
                        curr_lines.append(int(maybe_next.orig_line) if "orig_line" in transcript_df.columns else (idx2 + 1))
                        idx2 += 2
                        continue
                break
            break

        output_rows.append({"speaker": curr_speaker, "utterance": curr_text, "merge_lines": curr_lines})
        pbar.update(idx2 - idx)
        idx = idx2

    pbar.close()
    result_df = pd.DataFrame(output_rows)
    result_df.to_csv(OUTPUT_FILE_MERGE, index=False)
    print(f"Merged transcript at {OUTPUT_FILE_MERGE} with {len(result_df)} rows")

# run the scripts
if __name__ == "__main__":
    run_backchannel()
    run_merger()
