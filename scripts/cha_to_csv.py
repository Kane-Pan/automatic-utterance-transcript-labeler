# cha_to_csv.py
# Convert a CHAT (.cha) transcript into a CSV file with alternating utterances from speakers

import csv
from pathlib import Path 

# Paths
parent_dir = Path(__file__).parent.parent

input_file = parent_dir/"raw_data"/"CALLHOME_2.cha"
output_file = parent_dir/"data"/"CALLHOME_cleaned_2.csv"


def clean_utterance(utterance: str) -> str:
    i = 0
    out = []

    while i < len(utterance):
        ch = utterance[i]

        # Case 1: &=
        if ch == "&" and i + 1 < len(utterance) and utterance[i+1] == "=":
            # find next space (end of the tag)
            j = utterance.find(" ", i)
            if j == -1:
                j = len(utterance)
            # substring after '=' up to the next space
            tag = utterance[i+2:j]
            out.append("[" + tag.upper() + "]")
            i = j
            continue

        # Case 2: &-
        if ch == "&" and i + 1 < len(utterance) and utterance[i+1] == "-":
            # skip the "&-"
            i += 2
            continue

        # Default: copy character
        out.append(ch)
        i += 1

    return "".join(out)

# Open the .cha file and read all lines
with open(input_file, "r", encoding="utf-8", errors="ignore") as f:
    lines = f.readlines()

rows = [] 
speaker = ""  
utterance = "" 

for line in lines:
    line = line.strip()

    # Skip metadata lines (start with @)
    if line.startswith("@"):
        continue

    if "" in line:
        start = line.index("")
        end = line.index("", start + 1)
        line = line[:start] + line[end + 1:]

    # If line starts with "*", then it's a new speaker turn
    if line.startswith("*"):
        # Split speaker and text
        parts = line.split(":", 1)
        new_speaker = parts[0].strip("*")
        new_text = parts[1].strip()

        # If same speaker appears consecutively, merge into a single utterance
        if utterance != "" and new_speaker == speaker:
            utterance = (utterance + " " + new_text).strip()
        else:
            # Save previous utterance before starting new one
            if utterance != "":
                rows.append([speaker, utterance])
                utterance = ""
            speaker = new_speaker
            utterance = new_text

    # If line starts with tab then its a cont. utterance of the same speaker
    elif line.startswith("\t"):
        utterance += " " + line.strip()

    utterance = clean_utterance(utterance.strip())
    utterance = utterance.replace("xxx", "[UNINTELLIGIBLE]")

# add last utterance
if utterance != "":
    rows.append([speaker, utterance])

# Write to CSV
with open(output_file, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["speaker", "utterance"])
    writer.writerows(rows)

print("Saved", len(rows), "utterances to", output_file)
