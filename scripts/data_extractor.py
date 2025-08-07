import pandas as pd
import numpy as np
from pathlib import Path

# paths
BASE_DIR = Path(__file__).resolve().parent.parent
MANUAL_FILE = BASE_DIR/"data"/"manually_labeled_transcript.csv"
AUTO_FILE = BASE_DIR/"results"/"labeled_transcript.csv"
CLEANED_FILE = BASE_DIR/"results"/"labeled_cleaned_transcript.csv"
RESULTS_DIR = BASE_DIR/"results"

def read_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    for col in ["start", "stop"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def duration(df: pd.DataFrame) -> pd.Series:
    # Compute duration of each utterance in seconds
    return (df["stop"] - df["start"]).astype(float)


def compute_total_utterances_per_speaker(df: pd.DataFrame, speaker_col: str = "speaker") -> pd.Series:
    # Return a series mapping speaker IDs to total utterance counts
    return df[speaker_col].value_counts()


def compute_label_counts(df: pd.DataFrame, label_col: str, labels: list) -> dict:
    # compute label counts for a given column
    counts = {label: 0 for label in labels}
    value_counts = df[label_col].astype(str).value_counts(dropna=False)
    for label in labels:
        counts[label] = int(value_counts.get(label, 0))
    return counts


def compute_label_percentages(counts: dict, total: int) -> dict:
    # convert counts to percentages
    percentages = {}
    for label, count in counts.items():
        if total > 0:
            percentages[label] = 100 * count / total
        else:
            percentages[label] = np.nan
    return percentages


def compute_average_discrepancy(merged_df: pd.DataFrame, manual_col: str, auto_col: str) -> float:
    # Compute the average absolute difference between manual and automatic labels.
    discrepancies = []
    for _, row in merged_df.iterrows():
        m_label = str(row[manual_col])
        a_label = str(row[auto_col])
        # Skip if either label is not a digit (backchannel or unclassified)
        if not m_label.isdigit() or not a_label.isdigit():
            continue
        try:
            m_val = int(m_label)
            a_val = int(a_label)
            discrepancies.append(abs(m_val - a_val))
        except ValueError:
            # If conversion fails, skip this row
            continue
    if discrepancies:
        return float(np.mean(discrepancies))
    else:
        return float("nan")


def compute_average_label_per_speaker(df: pd.DataFrame, label_col: str, speaker_col: str = "speaker") -> pd.Series:
    # Compute the average numeric label per speaker.

    labels_df = df[[speaker_col, label_col]].copy()
    labels_df[label_col] = pd.to_numeric(labels_df[label_col], errors="coerce")
    return labels_df.groupby(speaker_col)[label_col].mean()


def main():
    out_dir = RESULTS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load CSVs
    manual_df = read_csv(MANUAL_FILE)
    auto_df = read_csv(AUTO_FILE)
    cleaned_df = read_csv(CLEANED_FILE)
    # Ensure the cleaned transcript has a column named 'auto_label'
    # Sometimes the cleaned file may have 'final_label' instead
    # treat 'final_label' as 'auto_label'
    if "auto_label" not in cleaned_df.columns:
        if "final_label" in cleaned_df.columns:
            cleaned_df = cleaned_df.rename(columns={"final_label": "auto_label"})


    merged = pd.merge(
        manual_df[["turn_id", "final_label"]].rename(columns={"final_label": "manual_label"}),
        auto_df[["turn_id", "final_label"]].rename(columns={"final_label": "auto_label"}),
        on="turn_id",
        how="inner"
    )

    # Compute average discrepancy ignoring backchannels
    avg_discrepancy = compute_average_discrepancy(merged, "manual_label", "auto_label")

    # Prepare automatically labeled transcript summary
    auto_total = len(auto_df)
    auto_counts = compute_label_counts(auto_df, "final_label", ["0", "1", "2", "3", "4", "5", "backchannel"])
    auto_percentages = compute_label_percentages(auto_counts, auto_total)
    auto_per_speaker = compute_total_utterances_per_speaker(auto_df)

    # Prepare cleaned transcript summary
    cleaned_total = len(cleaned_df)

    cleaned_counts = compute_label_counts(cleaned_df, "auto_label", ["0", "1", "2", "3", "4", "5"])
    cleaned_percentages = compute_label_percentages(cleaned_counts, cleaned_total)
    cleaned_per_speaker = compute_total_utterances_per_speaker(cleaned_df)
    # Average numeric label per speaker in cleaned transcript
    avg_label_per_speaker = compute_average_label_per_speaker(cleaned_df, "auto_label")

    # Write summary CSVs
    auto_summary_rows = []

    # Per speaker counts
    for speaker_id, count in auto_per_speaker.items():
        auto_summary_rows.append({"metric": f"total_turns_{speaker_id}", "value": count})

    # Label counts and percentages
    for label in ["0", "1", "2", "3", "4", "5", "backchannel"]:
        auto_summary_rows.append({"metric": f"label_count_for_{label}", "value": auto_counts.get(label, 0)})
        auto_summary_rows.append({"metric": f"percentt_label_{label}", "value": round(auto_percentages.get(label, np.nan), 2)})
   
    # Add overall total
    auto_summary_rows.append({"metric": "total_turns", "value": auto_total})

    # Add average discrepancy between manual and auto labels
    auto_summary_rows.append({"metric": "avg_label_discrepancy", "value": round(avg_discrepancy, 2) if not np.isnan(avg_discrepancy) else np.nan})
    auto_summary_df = pd.DataFrame(auto_summary_rows)
    auto_summary_df.to_csv(out_dir/"auto_label_metrics_summary.csv", index=False)


    cleaned_summary_rows = []

    for speaker_id, count in cleaned_per_speaker.items():
        cleaned_summary_rows.append({"metric": f"total_turns_{speaker_id}", "value": count})

    for label in ["0", "1", "2", "3", "4", "5"]:
        cleaned_summary_rows.append({"metric": f"label_count_for_{label}", "value": cleaned_counts.get(label, 0)})
        cleaned_summary_rows.append({"metric": f"percent_label_{label}", "value": round(cleaned_percentages.get(label, np.nan), 2)})
 
    cleaned_summary_rows.append({"metric": "total_turns", "value": cleaned_total})

    for speaker_id, avg_label in avg_label_per_speaker.items():
        cleaned_summary_rows.append({"metric": f"avg_label_{speaker_id}", "value": round(avg_label, 2) if not np.isnan(avg_label) else np.nan})
    cleaned_summary_df = pd.DataFrame(cleaned_summary_rows)
    cleaned_summary_df.to_csv(out_dir/"cleaned_label_metrics_summary.csv", index=False)

    # Identify mismatches between manual and auto labels for reference
    mismatches = merged[
        merged["manual_label"].astype(str).str.lower() != merged["auto_label"].astype(str).str.lower()].copy()
    mismatches["error_type"] = (
        mismatches["auto_label"].astype(str).str.lower() + "->" + mismatches["manual_label"].astype(str).str.lower()
    )
    mismatch_cols = ["turn_id", "auto_label", "manual_label", "error_type"]
    mismatches[mismatch_cols].to_csv(out_dir/"label_discrepancies.csv", index=False)

    print("Finished creating label statistics.")
    print("Auto metrics file: ", out_dir/"auto_label_metrics_summary.csv")
    print("Cleaned metrics file: ", out_dir/"cleaned_label_metrics_summary.csv")
    print("Mismatch file: ", out_dir/"label_discrepancies.csv")


if __name__ == "__main__":
    main()