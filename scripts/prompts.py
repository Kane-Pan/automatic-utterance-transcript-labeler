prompt_list = {
    # Kendall's Tau: 0.3667, p-value: 2.416e-23
    # Cohen's Kappa: 0.3655
    "soc_1" : """
You are a strict classifier for conversational utterances. You must assign one integer label from 1 to 5 based on the amount of social effort required to make the utterance.

**LABELING SCALE:**
1 – The utterance requires little or no social effort, usually applies to very common lines people automatically say in conversations without needing to think a lot.
5 – The utterance shows that the speaker is highly engaged in the conversation. The utterance could be personalized, often opinionated or providing an explanation, context-specific content, showing high cognitive and social effort. 

**RULES:**
- Output **exactly one** integer: **0**, **1**, **2**, **3**, **4**, or **5**.  
- Do **NOT** add any other text, punctuation, or justification.
- Do not be reserved in labelling utterances as **1** or **5**. Label as such if you think the utterance is very low or very high effort.

Now, classify the CURRENT utterance based on the previous utterance and the current utterance:
Previous utterance: "{prev_utt}"  
Current utterance to classify: "{cur_utt}"
""",

    # Kendall's Tau: 0.3523, p-value: 4.522e-22
    # Cohen's Kappa: 0.3753
    # Note: heavy leaning towards 5s
    "soc_2" : """
You are a strict classifier for conversational utterances. You must assign one integer label from 1 to 5 based on the amount of cognitive and social effort required to make the utterance.

**LABELING SCALE:**
1 – The utterance requires little or  no cognitive effort, usually applies to very common lines people automatically say in conversations without thinking
5 – The utterance shows that the speaker is highly engaged in the conversation. The utterance could be personalized, often opinionated or providing an explanation, context-specific content, showing high cognitive and social effort. 

**RULES:**
- Output **exactly one** integer: **0**, **1**, **2**, **3**, **4**, or **5**.  
- Do **NOT** add any other text, punctuation, or justification.
- Do not be reserved in labelling utterances as **1** or **5**. Label as such if you think the utterance is very low or very high effort.

**Context:**  The following utterances are excerpted from a conversation between two speakers who are speaking online via Zoom during the COVID-19 pandemic.

Now, classify the CURRENT utterance based on the previous utterance and the current utterance:
Previous utterance: "{prev_utt}"  
Current utterance to classify: "{cur_utt}"
    """,

    # Kendall's Tau: 0.0113, p-value: 0.7701
    # Cohen's Kappa: 0.0151
    # Note: AI generated prompt from ChatGPT-5. Results as expected, strong adherence to the scale criteria and returns only 2s and 3s.
    "soc_3" : """You are a grader. Output exactly ONE of these digits: 1,2,3,4,5. Do not output anything else.

SCALE (decide by the CURRENT utterance; use Previous only for context):
1 = rote/phatic or formulaic social talk with minimal content (e.g., thanks, OK, sure, greeting, small talk).
2 = brief, context-light reply or request; minimal information (yes/no/maybe; short choices; quick acknowledgement with a small detail).
3 = simple informational or mild opinion with little/no justification (one clause; short fact, time, or preference).
4 = specific, multi-clause, or causal explanation with some details, references, or planning.
5 = highly engaged: personalized, explicit reasoning or justification, multi-step explanation, synthesis, or planning with rationale.

Rules:
- If uncertain between two adjacent labels, choose the LOWER one.
- Do not use 5 unless clear evidence of explanation/justification is present.
- Ignore leading discourse markers (“yeah”, “well”, “so”) and grade the substantive part only.
- Very short utterances (≤2 words) that are not backchannels are usually 1–2 unless they carry a unique fact (e.g., a number or name).

Previous: "{prev_utt}"
Current: "{cur_utt}"
Answer:
""",

# Kendall's Tau: 0.0995, p-value: 0.008617
# Cohen's Kappa: 0.0853
"soc_4" : """
You grade the SOCIAL EFFORT of the CURRENT utterance on a 1–5 scale.

Definition (intuitive): social effort = the speaker’s adaptation to the partner and situation — perspective-taking, responsiveness, inference, empathy, or added reasoning beyond scripted phrases.

Output rule:
- Output exactly ONE digit: 1, 2, 3, 4, or 5. No other text.

General guidance (soft, not rigid rules):
- Consider the PREVIOUS utterance only to judge how tailored the CURRENT line is.
- Ignore fillers and politeness padding (“uh”, “yeah”, “well”, “so”, “please”, “thanks”) and grade the substantive intent.
- Use the whole range. If torn between two numbers, choose the LOWER one (avoid defaulting to the middle).

Heuristic (additive, flexible):
1) Start from 2 (a minimally informative, non-personal reply).
2) Add +1 for each clear signal present:
   • Tailors to the partner’s specific content/state (direct follow-up, referencing their issue or details).
   • Shows inference/empathy/stance toward the partner (acknowledging feelings or implications).
   • Adds reasoning, explanation, planning, or specific detail beyond a quick reply.
3) Drop to 1 if it’s a generic script/phatic line (e.g., rote greeting, receipt-like phrase).
4) Cap at 5. Reserve 5 for cases with multiple clear signals of engagement.

Few-shot calibrators:
Prev: "" 
Curr: "Hi, how are you?"  → 1

Prev: "I sprained my ankle."
Curr: "How are you doing?"  → 3

Prev: "I sprained my ankle."
Curr: "Ouch—that sounds rough; are you able to walk today?"  → 4

Prev: "Where should we meet?"
Curr: "Starbucks at 3 on Main."  → 3

Prev: "Why did the script fail?"
Curr: "Because the DB restarted; I’ll rerun after cache warmup."  → 5

Now grade this pair:
Previous: "{prev_utt}"
Current: "{cur_utt}"
Answer:
"""

}
