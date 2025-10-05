# best_per_wp.py
# Streams all combinations of (wp, n1, Pnd, Np1, Helix),
# computes bending stress and % diff to 'sat',
# and keeps ONLY the single best (smallest |%diff|) per wp (input RPM).

import itertools, math, csv, time, numpy
import calculations as c
import functions as fn

# target (allowable) for % diff calc
sat = 36.8403
sac = 129.242

# -------------------------
# SEARCH SPACE
# -------------------------
VARIABLES = {
    # wp: input rpm (integers) 1200..3600 inclusive step 200
    "wp":   range(1200, 3601, 100),

    # n1: stage 1 ratio (floats)
    "n1":   numpy.arange(1, 9.1, 0.1),

    # Pnd: normal diametral pitch (list of discrete choices)
    "Pnd":  [4, 5, 6, 8, 10],

    # Np1: pinion teeth
    "Np1":  range(10, 101, 1),

    # Helix: degrees (floats)
    "Helix": [15, 20, 25],
}

OUT_CSV = "best_per_wp.csv"

def main():
    names   = list(VARIABLES.keys())
    domains = [list(VARIABLES[k]) for k in names]
    combos  = itertools.product(*domains)

    # best result per wp: { wp: {**params, "sigma_bend":..., "percent_diff":..., "abs_percent_diff":..., ...} }
    best_per_wp = {}

    checked = 0
    t0 = time.time()

    for values in combos:
        checked += 1
        params = dict(zip(names, values))
        wp = params["wp"]

        sigma_b = c.bending_stress(**params)
        sigma_c = c.contact_stress(**params)

        pdiff_b = fn.distance(sigma_b, sat)
        pdiff_c = fn.distance(sigma_c, sac)

        # Combined metric (sum of absolute percent diffs)
        combined_metric = abs(pdiff_b) + abs(pdiff_c)

        # Only keep the best (lowest combined_metric) for each wp
        if (wp not in best_per_wp) or (combined_metric < best_per_wp[wp]["combined_metric"]):
            best_per_wp[wp] = {
                **params,
                "sigma_bend": sigma_b,
                "percent_diff": pdiff_b,
                "abs_percent_diff": abs(pdiff_b),
                "sigma_contact": sigma_c,
                "contact_percent_diff": pdiff_c,
                "abs_contact_percent_diff": abs(pdiff_c),
                "combined_metric": combined_metric
            }

        # Periodic progress print
        if checked % 100000 == 0:
            elapsed = time.time() - t0
            rate = checked / max(1.0, elapsed)
            best_metric = min([v['combined_metric'] for v in best_per_wp.values()]) if best_per_wp else float('nan')
            print(f"checked={checked:,}  rate={rate:,.0f}/s  best_combined_metric={best_metric:.6f}")

    elapsed = time.time() - t0

    # -------------------------
    # WRITE RESULTS
    # -------------------------
    with open(OUT_CSV, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(names + [
            "sigma_bend", "bending_percent_diff", "abs_bending_percent_diff",
            "sigma_contact", "contact_percent_diff", "abs_contact_percent_diff",
            "combined_metric"
        ])
        for wp in sorted(best_per_wp):
            row = best_per_wp[wp]
            output = []
            for k in names:
                if k == "n1":
                    output.append(f"{row[k]:.1f}".rstrip('0').rstrip('.') if '.' in f"{row[k]:.1f}" else f"{row[k]:.1f}")
                else:
                    output.append(row[k])
            output += [
                row["sigma_bend"], row["percent_diff"], row["abs_percent_diff"],
                row["sigma_contact"], row["contact_percent_diff"], row["abs_contact_percent_diff"],
                row["combined_metric"]
            ]
            w.writerow(output)

    print(f"\nDone. Checked {checked:,} combinations in {elapsed:.2f}s "
          f"({checked/elapsed:,.0f} combos/s).")
    print(f"Wrote best row for each wp ({len(best_per_wp)} values) to {OUT_CSV}")

if __name__ == "__main__":
    main()
