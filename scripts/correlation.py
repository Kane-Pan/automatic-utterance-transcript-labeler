import pandas as pd
import numpy as np
from scipy.stats import kendalltau
from sklearn.metrics import cohen_kappa_score, confusion_matrix
from pathlib import Path
import krippendorff

# Load data
BASE_DIR = Path(__file__).resolve().parent.parent

# n CSV paths (add more if desired)
csv_files = [
    BASE_DIR/"data"/"CALLHOME_TEST_Akash.csv",
    BASE_DIR/"data"/"CALLHOME_TEST_Itamar.csv",
    BASE_DIR/"data"/"CALLHOME_TEST_Kane.csv",
    BASE_DIR/"data"/"CALLHOME_TEST_AI.csv"
]

# names = [name = Path(path).stem.replace("CALLHOME_TEST_", "") for path in csv_files]

# Choose "info_score" or "relationship_score"
LABEL_COL = "relational_score"
LABELS = [1, 2, 3, 4, 5]

# Merge all raters on row_id (since new CSVs lack turn_id)
merged = None
rater_cols = []

for path in csv_files:
    df = pd.read_csv(path)
    name = Path(path).stem.replace("CALLHOME_TEST_", "")
    col_name = f"{LABEL_COL}_{name}"

    # create a stable row index to align rows across files
    df = df.reset_index().rename(columns={"index": "row_id"})
    df = df[['row_id', LABEL_COL]].rename(columns={LABEL_COL: col_name})

    if merged is None:
        merged = df
    else:
        merged = pd.merge(merged, df, on='row_id', how='inner')

    rater_cols.append(col_name)

# Clean to numeric; blank becomes NaN
for col in rater_cols:
    merged[col] = merged[col].astype(str).str.strip()
    merged[col] = merged[col].replace({'': np.nan, 'nan': np.nan})
    merged[col] = pd.to_numeric(merged[col], errors='coerce')

# Krippendorff's alpha across all raters
if krippendorff is not None:
    reliability_data = merged[rater_cols].to_numpy().T
    alpha_ord = krippendorff.alpha(reliability_data=reliability_data, level_of_measurement='ordinal')
else:
    alpha_ord = np.nan

print(f"\n{LABEL_COL}: Multi-rater agreement:")
print(f"Krippendorff's alpha (ordinal): {alpha_ord if not np.isnan(alpha_ord) else 'NA (install `krippendorff`)'}")

# Pairwise Kendall's tau-b and quadratic-weighted Cohen's kappa
pairwise = []
i = 0
while i < len(rater_cols):
    j = i + 1
    while j < len(rater_cols):
        a_col = rater_cols[i]
        b_col = rater_cols[j]

        sub = merged[[a_col, b_col]].dropna()
        a = pd.to_numeric(sub[a_col], errors='coerce')
        b = pd.to_numeric(sub[b_col], errors='coerce')
        mask = (~a.isna()) & (~b.isna())
        a = a[mask]
        b = b[mask]

        if len(a) == 0:
            print(f"\n{a_col} vs {b_col}: no overlapping labeled items.")
            j += 1
            continue

        tau = kendalltau(a, b, nan_policy='omit')  # τ-b
        kappa = cohen_kappa_score(a, b, labels=LABELS, weights='quadratic')

        pairwise.append({
            "rater_a": a_col,
            "rater_b": b_col,
            "n_items": int(len(a)),
            "kendall_tau_b": float(tau.statistic),
            "kendall_p": float(tau.pvalue),
            "cohen_kappa_quadratic": float(kappa),
        })

        # Confusion matrices
        C = confusion_matrix(b, a, labels=LABELS)
        print(f"\nConfusion matrix for {a_col} vs {b_col}:")
        print(pd.DataFrame(C,index=[f"{b_col}_{l}" for l in LABELS], columns=[f"{a_col}_{l}" for l in LABELS]))

        j += 1
    i += 1

# Print summary
if pairwise:
    print(f"\n{LABEL_COL}: Pairwise agreement =")
    for r in pairwise:
        print(
            f"{r['rater_a']} vs {r['rater_b']} Stats: "
            f"n={r['n_items']}, "
            f"τ-b={r['kendall_tau_b']:.4f} (p={r['kendall_p']:.4g}), "
            f"κ_quad={r['cohen_kappa_quadratic']:.4f}"
        )
else:
    print("\nNo pairwise results to report.")