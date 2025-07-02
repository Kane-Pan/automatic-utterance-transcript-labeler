# Automatic vs Reflective Classifier for Natural Conversation Utterances

This repo contains a **two-pass pipeline** that labels each utterance in a natural conversation Candor
transcript as:

| Label | Meaning |
|-------|---------|
| **1** | Automatic |
| **2** | Reflective |
| **3** | Uncertain → requires manual review |

The classifier uses the open-source model **Mistral-7B-Instruct** served
locally via **[Ollama](https://ollama.com/)**.  
It runs entirely on your machine (GPU or CPU).

---

## Quick-start

```bash
# 1) create and activate the conda environment
conda env create -f environment.yml
conda activate candor_llm

# 2) download the model and start Ollama
ollama pull mistral:7b-instruct
ollama serve &     # starts REST API on http://localhost:11434

# 3) set desired output locations
Open scripts/auto_vs_reflect_labeler.py and edit the following variables:
transcript_path   # path for LLM-labeled output with ollama's reasoning (CSV)
save_path         # path for final numeric labels (CSV)
counts_save_path  # path for quicl sumnmary on label stats (txt)

# 4) run the two-pass labeller on the sample file
python scripts/auto_vs_reflect_labeler.py

