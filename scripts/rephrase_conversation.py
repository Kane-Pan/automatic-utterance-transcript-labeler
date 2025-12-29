import json
import re
import time
import requests
import pandas as pd

INPUT_CSV  = r"./data/CALLHOME_cleaned_2.csv"
OUTPUT_CSV = r"./data/CALLHOME_rephrased_info.csv"

MODE  = "info"   # "info" | "relational" | "clarity"
MODEL = "mistral:instruct"
OLLAMA_URL = "http://localhost:11434/api/generate"


NUM_CTX = 8192
TEMPERATURE = 0.2
SEED = 42

# If a full one-shot request fails, fall back to chunking
CHUNK_SIZE = 70
OVERLAP = 10

# -------------------- PROMPTS --------------------
def build_system_rules(mode: str) -> str:
    common = """
You rewrite utterances. You must follow these rules strictly:

1) Do NOT add new facts, names, events, or claims.
   Only use content already present or directly inferable from the original.
2) Keep the syntactic structure as similar as possible (same general clause order).
3) Preserve speaker intent and meaning (unless the task explicitly changes phrasing style).
4) Keep placeholders like [UNINTELLIGIBLE] as-is.
5) Output ONLY valid JSON, nothing else.

You will receive a transcript as numbered lines:
  <index>. <SPEAKER>: <UTTERANCE>

Return JSON exactly in this form:
{"rephrased": ["...", "...", ...]}

The list length must equal the number of input lines, in the same order.
"""
    if mode == "info":
        specific = """
Goal: maximize INFORMATION TRANSFER.
- Make vague language more precise.
- Turn casual check-ins into explicit, information-seeking wording when appropriate.
- Do not introduce any new information; just make the existing intent more explicit.

Example:
"how are you doing?" -> "what is your current physical and mental state?"
"""
    elif mode == "relational":
        specific = """
Goal: maximize RELATIONSHIP BUILDING.
- Increase warmth, empathy, politeness, and social connection.
- Keep the underlying meaning the same.
- Do not add new factual content; you can add social/affective framing only if it is consistent.

Example:
"how are you doing?" -> "hey, how are you feeling today? i’m glad to hear from you."
(only if consistent with the original tone; do not fabricate events.)
"""
    elif mode == "clarity":
        specific = """
Goal: CONTROL condition = rephrase for CLARITY ONLY.
- Keep meaning the same; fix transcription roughness and make it readable.
- Do not intensify info-transfer or relational tone beyond what is already there.

Example:
"uh i- i dunno" -> "i don’t know."
"""
    else:
        raise ValueError("MODE must be: info, relational, or clarity")

    return common.strip() + "\n\n" + specific.strip()

def build_one_shot_prompt(lines_text: str, mode: str) -> str:
    rules = build_system_rules(mode)
    prompt = f"""{rules}

Transcript to rewrite:
{lines_text}
"""
    return prompt

def build_chunk_prompt(context_text: str, chunk_text: str, mode: str) -> str:
    rules = build_system_rules(mode)
    prompt = f"""{rules}

Context (previous lines for coherence; DO NOT rewrite these):
{context_text}

Lines to rewrite (ONLY rewrite these):
{chunk_text}
"""
    return prompt

# -------------------- OLLAMA CALL --------------------
def call_ollama(prompt: str) -> str:
    payload = {
        "model": MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": TEMPERATURE,
            "seed": SEED,
            "num_ctx": NUM_CTX
        }
    }

    r = requests.post(OLLAMA_URL, json=payload, timeout=600)
    r.raise_for_status()

    data = r.json()
    # Ollama returns the model output under "response"
    return data.get("response", "")

# -------------------- JSON EXTRACTOR --------------------
def extract_json_obj(text: str):
    """
    Ollama outputs can include stray text. We extract the first JSON object/array we see.
    """
    text = text.strip()

    # Try direct parse first
    try:
        return json.loads(text)
    except Exception:
        pass

    # Try to find a JSON object inside
    m = re.search(r'(\{.*\})', text, flags=re.DOTALL)
    if m:
        candidate = m.group(1).strip()
        try:
            return json.loads(candidate)
        except Exception:
            pass

    # Try to find a JSON array
    m = re.search(r'(\[.*\])', text, flags=re.DOTALL)
    if m:
        candidate = m.group(1).strip()
        try:
            return json.loads(candidate)
        except Exception:
            pass

    return None

# ---------------- MAIN CODING LOGIC -------------
def make_lines(df: pd.DataFrame) -> list:
    lines = []
    for i in range(len(df)):
        spk = str(df.loc[i, "speaker"]) if "speaker" in df.columns else "SPEAKER"
        utt = str(df.loc[i, "utterance"]) if "utterance" in df.columns else ""
        lines.append(f"{i+1}. {spk}: {utt}")
    return lines

def one_shot_rephrase(lines: list, mode: str) -> list:
    lines_text = "\n".join(lines)
    prompt = build_one_shot_prompt(lines_text, mode)
    out = call_ollama(prompt)

    obj = extract_json_obj(out)
    if not obj or "rephrased" not in obj:
        raise RuntimeError("Model did not return expected JSON with key 'rephrased'.")

    rephrased = obj["rephrased"]
    if not isinstance(rephrased, list) or len(rephrased) != len(lines):
        raise RuntimeError(f"Expected {len(lines)} outputs, got {len(rephrased) if isinstance(rephrased, list) else 'non-list'}.")

    return rephrased

def chunked_rephrase(lines: list, mode: str, chunk_size: int, overlap: int) -> list:
    n = len(lines)
    results = []

    start = 0
    while start < n:
        end = start + chunk_size
        if end > n:
            end = n

        # context window = previous overlap lines
        ctx_start = start - overlap
        if ctx_start < 0:
            ctx_start = 0
        context_lines = lines[ctx_start:start]
        chunk_lines = lines[start:end]

        context_text = "\n".join(context_lines) if len(context_lines) > 0 else "(none)"
        chunk_text = "\n".join(chunk_lines)

        prompt = build_chunk_prompt(context_text, chunk_text, mode)
        out = call_ollama(prompt)

        obj = extract_json_obj(out)
        if not obj or "rephrased" not in obj:
            raise RuntimeError(f"Chunk {start}:{end} did not return expected JSON.")

        rephrased_chunk = obj["rephrased"]
        if not isinstance(rephrased_chunk, list) or len(rephrased_chunk) != len(chunk_lines):
            raise RuntimeError(f"Chunk {start}:{end} length mismatch: expected {len(chunk_lines)}, got {len(rephrased_chunk) if isinstance(rephrased_chunk, list) else 'non-list'}")

        results.extend(rephrased_chunk)

        print(f"Done chunk {start+1}-{end} / {n}")
        start = end
        time.sleep(0.05)

    return results

def main():
    df = pd.read_csv(INPUT_CSV)

    if "utterance" not in df.columns:
        raise ValueError("CSV must contain an 'utterance' column.")
    if "speaker" not in df.columns:
        print("Warning: no 'speaker' column found; continuing anyway.")

    lines = make_lines(df)
    out_col = f"utterance_rephrased_{MODE}"

    print(f"Loaded {len(df)} rows from {INPUT_CSV}")
    print(f"Mode = {MODE}, Model = {MODEL}, num_ctx = {NUM_CTX}")

    # Try one-shot first
    try:
        rephrased = one_shot_rephrase(lines, MODE)
        print("One-shot rewrite succeeded.")
    except Exception as e:
        print("One-shot rewrite failed. Falling back to chunked rewrite.")
        print("Reason:", str(e))
        rephrased = chunked_rephrase(lines, MODE, CHUNK_SIZE, OVERLAP)
        print("Chunked rewrite succeeded.")

    df[out_col] = rephrased
    df.to_csv(OUTPUT_CSV, index=False)
    print("Saved:", OUTPUT_CSV)

if __name__ == "__main__":
    main()
