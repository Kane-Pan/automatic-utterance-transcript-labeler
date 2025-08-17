import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import cohen_kappa_score, confusion_matrix, ConfusionMatrixDisplay
from scipy.stats import kendalltau
from pathlib import Path

# Load data and paths
BASE_DIR = Path(__file__).resolve().parent.parent
MANUAL_FILE = BASE_DIR/"data"/"manually_labeled_transcript.csv"
AUTO_FILE = BASE_DIR/"results"/"labeled_transcript.csv"

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

tau = kendalltau(auto, manual, nan_policy='omit')
kappa = cohen_kappa_score(auto, manual, labels=[0, 1, 2, 3, 4, 5], weights='quadratic')
C = confusion_matrix(manual, auto, labels=[0, 1, 2, 3, 4, 5])

print(f"Kendall's Tau: {tau.statistic:.4f}, p-value: {tau.pvalue:.4g}")
print(f"Cohen's Kappa: {kappa:.4f}")

disp = ConfusionMatrixDisplay(confusion_matrix=C, display_labels=[0, 1, 2, 3, 4, 5])
disp.plot(values_format='d')
plt.title("Manual vs AI Utterance Labels (Counts)")
plt.xlabel("AI-predicted label")
plt.ylabel("Manually assigned label")
plt.tight_layout()
plt.show()

C_norm = confusion_matrix(manual, auto, labels=[0, 1, 2, 3, 4, 5], normalize='true')
disp_norm = ConfusionMatrixDisplay(confusion_matrix=C_norm, display_labels=[0, 1, 2, 3, 4, 5])
disp_norm.plot(values_format='.2f')
plt.title("Manual vs AI Utterance Labels (Row-Normalized)")
plt.xlabel("AI-predicted label")
plt.ylabel("Manually assigned label")
plt.tight_layout()
plt.show()