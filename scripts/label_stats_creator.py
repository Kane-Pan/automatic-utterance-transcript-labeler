import pandas as pd
import numpy as np
from pathlib import Path

# paths
BASE_DIR = Path(__file__).resolve().parent.parent
MANUAL_FILE = BASE_DIR/"data"/"manually_labeled_transcript.csv"
AUTO_FILE = BASE_DIR/"results"/"labeled_transcript.csv"
CLEANED_FILE = BASE_DIR/"results"/"labeled_cleaned_transcript.csv"
RESULTS_DIR = BASE_DIR/"results"


def read_csv(path) -> pd.DataFrame:
    df = pd.read_csv(path)
    for col in ["start", "stop"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col])
    return df


def duration(df) -> pd.Series:
    return (df["stop"] - df["start"]).astype(float)


def avg_duration(cleaned_df, label_name) -> float:
    rows = cleaned_df[cleaned_df["auto_label"].str.lower() == label_name]
    if not rows.empty:
        return rows["duration"].mean()
    else:
        # If there are no rows for this label, return NaN
        return np.nan


def main():
    out_dir = RESULTS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load CSVs
    manual_df = read_csv(MANUAL_FILE)
    auto_df = read_csv(AUTO_FILE)
    cleaned_df = read_csv(CLEANED_FILE)

    merged = pd.merge(
        manual_df[["turn_id","utterance","final_label"]].rename(columns={"final_label":"manual_label"}),
        auto_df[["turn_id","final_label"]].rename(columns={"final_label":"auto_label"}),
        on="turn_id",
        how="inner"
    )

    # Find incorrect labels
    mismatches = merged[merged["manual_label"].str.lower()
                       != merged["auto_label"].str.lower()].copy()

    mismatches["error_type"] = (
        mismatches["auto_label"].str.lower() + "->" + mismatches["manual_label"].str.lower()
    )
    mismatch_cols = ["turn_id","utterance","auto_label","manual_label","error_type"]
    mismatches[mismatch_cols].to_csv(out_dir/"label_discrepancies.csv", index=False)

    # Compute durations and averages
    auto_df["duration"] = duration(auto_df)
    bc_auto = auto_df[auto_df["final_label"].str.lower() == "backchannel"]
    if not bc_auto.empty:
        avg_bc_duration = bc_auto["duration"].mean()
    else:
        avg_bc_duration = np.nan

    cleaned_df["duration"] = duration(cleaned_df)

    avg_auto_duration = avg_duration(cleaned_df, "automatic")
    avg_nonauto_duration = avg_duration(cleaned_df, "non-automatic")

    # Compute percentages in cleaned_df
    total_turns = len(cleaned_df)
    label_counts = cleaned_df["auto_label"].str.lower().value_counts(dropna=False)

    if total_turns > 0:
        pct_backchannel = 100 * (auto_df["final_label"].str.lower() == "backchannel").sum() / len(auto_df)
        pct_auto = 100 * label_counts.get("automatic", 0) / total_turns
        pct_nonauto = 100 * label_counts.get("non-automatic", 0) / total_turns
    else:
        pct_backchannel = np.nan
        pct_auto = np.nan
        pct_nonauto = np.nan

    # Build summary table
    summary = pd.DataFrame({
        "metric": [
            "avg_backchannel_duration_sec (raw_transcript)",
            "avg_automatic_duration_sec (cleaned_transcript)",
            "avg_nonautomatic_duration_sec (cleaned_transcript)",
            "%_backchannel (raw_transcript)",
            "%_automatic (cleaned_transcript)",
            "%_non_automatic (cleaned_transcript)",
            "total_turns (cleaned_transcript)"
        ],
        "value": [
            round(avg_bc_duration, 2),
            round(avg_auto_duration, 2),
            round(avg_nonauto_duration, 2),
            round(pct_backchannel, 2),
            round(pct_auto, 2),
            round(pct_nonauto, 2),
            total_turns
        ]
    })

    summary.to_csv(out_dir/"label_metrics_summary.csv", index=False)

    print("Finished creating label statistics.")
    print("- Mismatch file: ", out_dir/"label_discrepancies.csv")
    print("- Metrics file: ", out_dir/"label_metrics_summary.csv")


if __name__ == "__main__":
    main()
