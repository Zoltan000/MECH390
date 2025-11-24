"""
Combine generated random data with brute-force best data.

Outputs a unified CSV with columns:
wp,wf,P,n1,Pd1,Np1,Helix1,Pd2,Np2,Helix2,volume,s1_bending,s1_contact,s2_bending,s2_contact,valid,is_best

Usage:
    python combine_datasets.py --generated generated_data.csv --brute brute_force_merged.csv --out combined_data.csv
"""

import argparse
import pandas as pd
import numpy as np

ALLOWABLE_BENDING = 36.8403
ALLOWABLE_CONTACT = 129.242

TARGET_COLS = [
    "wp",
    "wf",
    "P",
    "n1",
    "Pd1",
    "Np1",
    "Helix1",
    "Pd2",
    "Np2",
    "Helix2",
    "volume",
    "s1_bending",
    "s1_contact",
    "s2_bending",
    "s2_contact",
    "valid",
    "is_best",
]


def load_generated(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()
    df = df.copy()
    df["is_best"] = 0
    return df


def load_brute(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()
    # Drop extra stage 3 cols if present
    keep = [
        "wp",
        "wf",
        "P",
        "n1",
        "Pd1",
        "Np1",
        "Helix1",
        "Pd2",
        "Np2",
        "Helix2",
        "volume",
        "s1_bending",
        "s1_contact",
        "s2_bending",
        "s2_contact",
    ]
    df = df[keep].copy()
    # Compute validity
    def is_valid(row):
        try:
            return (
                row["s1_bending"] < ALLOWABLE_BENDING
                and row["s2_bending"] < ALLOWABLE_BENDING
                and row["s1_contact"] < ALLOWABLE_CONTACT
                and row["s2_contact"] < ALLOWABLE_CONTACT
            )
        except Exception:
            return False
    df["valid"] = df.apply(is_valid, axis=1).astype(int)
    df["is_best"] = 1
    return df


def parse_args():
    ap = argparse.ArgumentParser(description="Combine generated and brute-force datasets.")
    ap.add_argument("--generated", type=str, default="generated_data.csv", help="Path to generated random data.")
    ap.add_argument("--brute", type=str, default="brute_force_merged.csv", help="Path to brute-force merged data.")
    ap.add_argument("--out", type=str, default="combined_data.csv", help="Output CSV.")
    return ap.parse_args()


def main():
    args = parse_args()
    gen = load_generated(args.generated)
    brute = load_brute(args.brute)
    combined = pd.concat([gen, brute], ignore_index=True)
    combined = combined[TARGET_COLS]
    # Drop rows with missing critical fields
    combined = combined.dropna(subset=TARGET_COLS)
    combined.to_csv(args.out, index=False)
    print(f"Wrote {len(combined)} rows to {args.out}")


if __name__ == "__main__":
    main()
