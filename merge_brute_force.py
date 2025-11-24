"""merge_brute_force.py

Merge all brute-force CSV result files in a folder into a single processed CSV.
Each source file is assumed to contain unique `wp` values (no overlaps). If
duplicates are found across files, a warning is emitted and the first occurrence
is kept.

Processing rules (same as prior script):
  - Drop columns if present: stages, n2, Pd3, Np3, Helix3
  - Strip whitespace from column names
  - Round to 2 decimals: wf, P, volume, s1_bending, s1_contact, s2_bending, s2_contact, s3_bending, s3_contact
  - Round to 1 decimal: n1
  - Ensure `wp` is integer (nullable Int64) and `volume` numeric

Usage examples (from project root directory):
  python merge_brute_force.py
  python merge_brute_force.py --in-dir brute_force_data --out brute_force_merged.csv
  python merge_brute_force.py --in-dir path/to/dir --pattern "wp_*.csv" --out merged.csv

Arguments:
  --in-dir   Directory containing brute-force CSV files (default: brute_force_data)
  --pattern  Glob pattern to select files within in-dir (default: "*.csv")
  --out      Output CSV file path (default: brute_force_merged.csv)
"""
from __future__ import annotations
from pathlib import Path
import argparse
import pandas as pd
import sys

DEFAULT_IN_DIR = "brute_force_data"
DEFAULT_PATTERN = "*.csv"
DEFAULT_OUT = "brute_force_merged.csv"

DROP_COLS = ["stages", "n2", "Pd3", "Np3", "Helix3"]
ROUND2 = [
    "wf", "P", "volume",
    "s1_bending", "s1_contact",
    "s2_bending", "s2_contact",
    "s3_bending", "s3_contact",
]


def process_df(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # Normalize column names
    df.columns = [c.strip() for c in df.columns]

    # Drop unwanted columns if present
    for c in DROP_COLS:
        if c in df.columns:
            df = df.drop(columns=[c])

    # Round selected columns
    for c in ROUND2:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").round(2)

    if "n1" in df.columns:
        df["n1"] = pd.to_numeric(df["n1"], errors="coerce").round(1)

    # Ensure numeric types
    if "wp" in df.columns:
        df["wp"] = pd.to_numeric(df["wp"], errors="coerce").astype(pd.Int64Dtype())
    if "volume" in df.columns:
        df["volume"] = pd.to_numeric(df["volume"], errors="coerce")

    return df


def merge_unique(files: list[Path]) -> pd.DataFrame:
    frames = []
    for f in files:
        try:
            df = pd.read_csv(f)
        except Exception as e:
            print(f"Failed to read {f}: {e}", file=sys.stderr)
            continue
        df_processed = process_df(df)
        df_processed["_source_file"] = f.name  # provenance (optional)
        frames.append(df_processed)

    if not frames:
        return pd.DataFrame()

    combined = pd.concat(frames, ignore_index=True, sort=False)

    if "wp" not in combined.columns:
        raise ValueError("Merged data must contain 'wp' column (none of the files had 'wp').")

    # Detect duplicates of wp
    dup_mask = combined["wp"].duplicated(keep="first")
    dup_count = dup_mask.sum()
    if dup_count > 0:
        print(f"Warning: {dup_count} duplicate wp values found. Keeping first occurrence.", file=sys.stderr)
        combined = combined[~dup_mask].copy()

    # Sort by wp
    combined = combined.sort_values(by=["wp"]).reset_index(drop=True)

    # Move wp to first column if needed
    cols = list(combined.columns)
    if cols[0] != "wp" and "wp" in cols:
        cols.remove("wp")
        cols = ["wp"] + cols
        combined = combined[cols]

    return combined


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Merge brute-force CSV result files into a single processed CSV")
    p.add_argument("--in-dir", default=DEFAULT_IN_DIR, help="Input directory containing CSV files")
    p.add_argument("--pattern", default=DEFAULT_PATTERN, help="Glob pattern for selecting files inside in-dir")
    p.add_argument("--out", default=DEFAULT_OUT, help="Output CSV file path")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    in_dir = Path(args.in_dir)
    if not in_dir.exists() or not in_dir.is_dir():
        print(f"Input directory not found or not a directory: {in_dir}", file=sys.stderr)
        sys.exit(1)

    files = sorted(in_dir.glob(args.pattern))
    if not files:
        print(f"No files matched pattern '{args.pattern}' in {in_dir}", file=sys.stderr)
        sys.exit(1)

    print(f"Merging {len(files)} files from {in_dir} -> {args.out}")

    try:
        merged = merge_unique(files)
    except Exception as e:
        print(f"Error during merge: {e}", file=sys.stderr)
        sys.exit(1)

    if merged.empty:
        print("Resulting merged DataFrame is empty; nothing written.", file=sys.stderr)
        sys.exit(1)

    out_path = Path(args.out)
    merged.to_csv(out_path, index=False)
    print(f"Merged CSV written: {out_path} (rows: {len(merged)})")


if __name__ == "__main__":
    main()
