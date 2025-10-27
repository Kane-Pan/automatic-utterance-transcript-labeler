import pandas as pd
import requests
import re
import io
import sys

INPUT_PATH = "data/CALLHOME_TEST.csv" # input csv file: columns speaker,utterance
OUTPUT_PATH = "results/cleaned_transcript.txt" # save raw LLM output as txt
OLLAMA_URL = "http://localhost:11434/api/chat"
MODEL_NAME = "mistral:instruct"  # Mistral 7B Instruct on Ollama
TEMPERATURE = 0

df = pd.read_csv(INPUT_PATH)

# Keep only the two columns and fill blanks
df = df[["speaker", "utterance"]].copy()
df["speaker"] = df["speaker"].fillna("")
df["utterance"] = df["utterance"].fillna("")

lines = []
line_number = 1
for idx in range(len(df)):
    s = str(df["speaker"].iloc[idx]).strip()
    u = str(df["utterance"].iloc[idx]).replace("\r\n", " ").replace("\n", " ").strip()
    line_text = "[" + str(line_number) + "] " + s + ": " + u
    lines.append(line_text)
    line_number += 1

enumerated_transcript = "\n".join(lines)

# Prompts
SYSTEM_PROMPT = (
    "You are a careful transcript cleaner and reformatter.\n\n"
    "Your ONLY output must be a CSV fenced block with header:\n"
    "speaker,utterance,merged_lines\n\n"
    "Rules:\n"
    "- Preserve meaning and chronology. Do NOT invent or paraphrase content.\n"
    "- Remove bracketed noise/static tokens like [DISTORTION], [UNINTELLIGIBLE], [NOISE], "
    "[CROSSTALK], [LAUGHTER] when they represent non-linguistic artifacts.\n"
    "- Remove obvious backchannels and filler lines from the other speaker (e.g., \"uh huh\", "
    "\"mm\", \"yeah\", \"right\", \"okay\", \"hmm\", \"uh\", \"mhm\").\n"
    "- If a speaker’s sentence is split by a brief interjection from the other speaker, MERGE "
    "the parts back into a single continuous line for the original speaker.\n"
    "- Keep speaker names exactly as provided. Do not rename or swap speakers.\n"
    "- Do not add or reorder content beyond these rules.\n\n"
    "Output format:\n"
    "- CSV with columns: speaker,utterance,merged_lines\n"
    "- Each output row is one cleaned/merged utterance.\n"
    "- merged_lines is a bracketed comma-separated list of the ORIGINAL line numbers (1-based) "
    "that formed this row, like [3] or [1,3,5].\n"
    "- Quote fields that contain commas, quotes, or newlines (RFC 4180).\n"
    "- No extra columns. No commentary outside the CSV fence.\n\n"
    "Example behavior:\n"
    "Input:\n"
    "[1] A: I was having a great day today because it was sunny\n"
    "[2] B: uh huh\n"
    "[3] A: and it was warm\n\n"
    "Desired output row:\n"
    "A,\"I was having a great day today because it was sunny and it was warm\",\"[1,3]\"\n"
)

USER_PROMPT = (
    "You will receive a transcript as numbered lines in the format:\n"
    "[LINE_NUMBER] SpeakerLetter: utterance\n\n"
    "Clean it according to the rules and return ONLY a CSV fenced block with the header:\n"
    "speaker,utterance,merged_lines\n\n"
    "Transcript:\n"
    "```\n" + enumerated_transcript + "\n```"
)

# Call Ollama
payload = {
    "model": MODEL_NAME,
    "messages": [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": USER_PROMPT}
    ],
    "options": {
        "temperature": 0,
        "num_ctx": 8192,
        "num_predict": 4096
    },
    "stream": False  # return to JSON
}

try:
    r = requests.post(OLLAMA_URL, json=payload, timeout=600)
except Exception as e:
    raise SystemExit("Could not reach Ollama at " + OLLAMA_URL + "and is it running: \n" + str(e))

if r.status_code != 200:
    raise SystemExit("Ollama returned HTTP:" + str(r.status_code) + ":\n" + r.text)

resp_json = r.json()

if "message" not in resp_json or "content" not in resp_json["message"]:
    raise SystemExit("Unexpected Ollama response:\n" + str(resp_json))

llm_text = resp_json["message"]["content"]

# output to a txt file
with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
    f.write(llm_text)

print("Wrote raw LLM output to", OUTPUT_PATH)
