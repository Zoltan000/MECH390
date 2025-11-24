"""
Random gearbox dataset generator using calculations.py and functions.py.

Generates combinations of gearbox parameters, computes stresses/volume,
and labels each sample as valid if all stresses are below allowable limits.

Outputs a CSV with columns:
    wp, wf, P, n1, Pd1, Np1, Helix1, n2, Pd2, Np2, Helix2,
    volume, s1_bending, s1_contact, s2_bending, s2_contact, valid

Usage:
    python generate_gearbox_data.py --out data_generated.csv --rows 50000
"""

from __future__ import annotations

import argparse
import random
import csv
from typing import Optional, Tuple

import numpy as np

import calculations as calc


ALLOWABLE_BENDING = 36.8403  # ksi
ALLOWABLE_CONTACT = 129.242  # ksi

# Allowed discrete options
VALID_PDN = [4, 5, 6, 8, 10]
VALID_HELIX = [15, 20, 25]


def sample_parameters() -> Tuple[float, int, int, int, int, int, int]:
    """Sample a random set of gearbox parameters."""
    wp = random.uniform(1200.0, 3600.0)
    # wf is implied inside calculations.important_values; keep for record
    wf = wp / 12.0 + 100.0
    n1 = random.uniform(1.2, 6.0)
    Pd1 = random.choice(VALID_PDN)
    Pd2 = random.choice(VALID_PDN)
    Helix1 = random.choice(VALID_HELIX)
    Helix2 = random.choice(VALID_HELIX)
    Np1 = random.randint(12, 60)
    Np2 = random.randint(12, 60)
    return wp, wf, n1, Pd1, Np1, Helix1, Pd2, Np2, Helix2


def evaluate_sample(params) -> Optional[dict]:
    """Compute stresses and volume for a sampled parameter set; drop if calculation fails."""
    wp, wf, n1, Pd1, Np1, Helix1, Pd2, Np2, Helix2 = params
    try:
        wf_out, P_out, volume, s1_b, s1_c, s2_b, s2_c, _, _ = calc.results(
            wp,
            n1,
            None,  # let calculations derive n2
            Pd1,
            Np1,
            Helix1,
            Pd2,
            Np2,
            Helix2,
            None,
            None,
            None,
        )
    except Exception:
        return None

    if volume is None or np.isnan(volume):
        return None

    valid = (
        s1_b is not None
        and s2_b is not None
        and s1_c is not None
        and s2_c is not None
        and s1_b < ALLOWABLE_BENDING
        and s2_b < ALLOWABLE_BENDING
        and s1_c < ALLOWABLE_CONTACT
        and s2_c < ALLOWABLE_CONTACT
    )

    return {
        "wp": wp,
        "wf": wf_out,  # computed inside calculations
        "P": P_out,
        "n1": n1,
        "Pd1": Pd1,
        "Np1": Np1,
        "Helix1": Helix1,
        "n2": wp / wf_out / n1 if wf_out else None,
        "Pd2": Pd2,
        "Np2": Np2,
        "Helix2": Helix2,
        "volume": volume,
        "s1_bending": s1_b,
        "s1_contact": s1_c,
        "s2_bending": s2_b,
        "s2_contact": s2_c,
        "valid": int(valid),
    }


def generate_rows(num_rows: int):
    """Yield generated rows until num_rows collected (skips failures)."""
    collected = 0
    while collected < num_rows:
        params = sample_parameters()
        res = evaluate_sample(params)
        if res is None:
            continue
        collected += 1
        if collected % 5000 == 0:
            print(f"Generated {collected}/{num_rows}")
        yield res


def write_csv(path: str, num_rows: int) -> None:
    fieldnames = [
        "wp",
        "wf",
        "P",
        "n1",
        "Pd1",
        "Np1",
        "Helix1",
        "n2",
        "Pd2",
        "Np2",
        "Helix2",
        "volume",
        "s1_bending",
        "s1_contact",
        "s2_bending",
        "s2_contact",
        "valid",
    ]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in generate_rows(num_rows):
            writer.writerow(row)
    print(f"Wrote {num_rows} rows to {path}")


def parse_args():
    parser = argparse.ArgumentParser(description="Generate gearbox dataset.")
    parser.add_argument("--out", type=str, default="generated_data.csv", help="Output CSV path.")
    parser.add_argument("--rows", type=int, default=50000, help="Number of rows to generate.")
    return parser.parse_args()


def main():
    args = parse_args()
    write_csv(args.out, args.rows)


if __name__ == "__main__":
    main()
