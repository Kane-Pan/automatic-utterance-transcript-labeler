import pandas as pd
import requests
from tqdm import tqdm
from pathlib import Path

api_url = "http://localhost:11434/api/generate"
ollama_model = "mistral:7b-instruct"

def create_prompt_1(prev: str, cur: str) -> str:
    return f"""
    You are a classifier that labels conversational utterances as Automatic or Reflective.

Automatic utterances:
- Sound robotic or rehearsed, as if following a script
- Lone backchanneling cues (e.g. "I see", "right", "Yeah", "uh-huh", "Mhm")
- Common social openings/greetings/answers that does not rely on any unique context (e.g. "Hi, how are you?", "Where do you live?", "Good morning", "What’s up?, "My name is John")
- Closed questions that ask for mundane things unrelated to the speaker (e.g. "Isn't the weather nice today?", "It’s been a long week")

Reflective utterances:  
- Shows on-the-fly thinking, adaptation, or acknowledgement to the other person or current situation 
- Refers to unique details in response to the context of the conversation or environment (e.g. "I actually live in Alabama", "I like your shirt!")
- Asks open-ended questions or adds new information to invite elaboration
- Asks personal questions to learn about the other person (e.g. "What do you do for a living?", "What did you think of the game last night?", "Do you have any hobbies?")
- Expresses personal thoughts, feelings, opinions, or experiences

I want you to conduct a step-by-step analysis to classify the utterance as either "Automatic" or "Reflective".
You will follow these steps:
1. Analyze whether the current utterance shows scripted or thoughtful behavior.  
2. Compare features of this utterance to the definitions of Automatic vs. Reflective.  
3. Decide which label fits best.  
4. End every response with the label "Automatic" or "Reflective".

The following utterance is a line from a conversation between two people. Classify the CURRENT utterance as "Automatic" or "Reflectrive", given it is a response to the previous utterances:

Previous utterance from other speaker : "{prev}"
CURRENT utterance (to be labeled by you): "{cur}"
"""

# Example utterances for few-shot prompting if needed:
'''
"Uh-huh, that's right." Automatic
"Alright, let’s get started." Automatic
"what country are you from?" Automatic

"I was gonna say what made you leave Florida for Kentucky?" Reflective
"There’s, as far as current events goes, there’s a lot of really stupid people. Uh" Reflective
"How does it feel to talk to a stranger someone you never met?" Reflective
'''

# Function to call the ollama API with the prompt
def call_ollama(prompt_1: str) -> str:
    payload = {
        "model": ollama_model,
        "prompt": prompt_1,
        "max_tokens": 500,
        "temperature": 0.0,
        "stream": False
    }
    r = requests.post(api_url, json=payload)
    r.raise_for_status()
    return r.json().get("response", "").strip()


def second_call_ollama(labeled_csv_path: str) -> pd.DataFrame:
    """
    Reads the CSV file produced after the first labeling phase and re-classifies each
    descriptive label into a single digit (1,2,3)
    """
    df = pd.read_csv(labeled_csv_path)
    final_labels = []
    for desc in df["label"]:

        # preserving blanks
        if pd.isna(desc) or not str(desc).strip():
            final_labels.append(desc)
            continue

        number_labeling_prompt = f"""
The following is an explanation and verdict for one conversation utterance labeled as Automatic or Reflective:

{desc}

Now output exactly one digit 1, 2, or 3 based on the following criteria:
1 if the verdict is Automatic,
2 if the verdict is Reflective,
3 if you are uncertain or it is ambiguous.

DO NOT output any extra text. Just the digit.
"""
        resp = call_ollama(number_labeling_prompt).strip()
        if resp.isdigit() and resp in ["1", "2", "3"]:
            digit = int(resp)
        else:
            digit = 3 # default to uncertain if response is invalid

        final_labels.append(digit)
    df["label"] = final_labels
    return df

def count_labels(df: pd.DataFrame, col: str, save_location: str):
    """
    Counts the occurrences of each label 1, 2, 3, in the labeled csv.
    """
    if col not in df.columns:
        print(f"No {col} column found.")
        return

    labels = df[col].astype(int)
    total_lines = len(labels)

    ones = 0
    twos = 0
    threes = 0

    for label in labels:
        if label == 1:
            ones += 1
        elif label == 2:
            twos += 1
        elif label == 3:
            threes += 1

    with open(save_location, 'w') as file:
        file.write(
            f"Total lines: {total_lines}\n"
            f"Automatic: {ones}\t"
            f"Reflective: {twos}\t"
            f"Uncertain (requires manual review): {threes}\n"
            f"Percentage Automatic: {ones / total_lines * 100:.2f}%\t"
            f"Percentage Reflective: {twos / total_lines * 100:.2f}%\t"
            f"Percentage Uncertain: {threes / total_lines * 100:.2f}%\n"
        )

    return

def convert_labels(csv_path: str, output_path: str) -> pd.DataFrame:
    """
    Calls second_call_ollama on the first-pass CSV, writes
    out the numeric labels CSV, and returns the final DataFrame.
    """
    df_final = second_call_ollama(csv_path)
    df_final.to_csv(output_path, index=False)
    return df_final

def main():
    # file path for first-pass labeled transcript
    transcript_path = "DESIRED_CSV_FILE_NAME.csv"

    # file path for final numeric labels
    save_path = "Path/to/desired/save_location/file_name.csv"
    counts_save_path = "Path/to/desired/save_location/for/results/file_name.csv"

    if not Path(save_path).parent.is_dir():
        raise FileNotFoundError(f"Directory {Path(save_path).parent} does not exist.")

    # load raw transcript
    df = pd.read_csv("test_transcript.csv", usecols = ["turn_id","speaker","start","stop","utterance"])
    df["prev_utterance"] = df["utterance"].shift(1).fillna("")
    df["label"] = ""

    # first classification
    for index, row in tqdm(df.iterrows(), total=len(df)):
        prompt = create_prompt_1(row["prev_utterance"], row["utterance"])
        df.at[index, "label"] = call_ollama(prompt)

    df.to_csv(transcript_path, index=False)
    print(f"Labels (with reasoning) saved to {transcript_path}")

    # second-pass numeric conversion
    df_final = convert_labels(transcript_path, save_path)
    print(f"Final numeric labels (1, 2, 3) saved to {save_path}")

    # count and save label stats
    count_labels(df_final, "label", counts_save_path)
    print(f"Label stats saved to {counts_save_path}")

if __name__ == "__main__":
    main()