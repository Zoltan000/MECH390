# best_per_wp.py
# Streams all combinations of (wp, n1, Pnd, Np1, Helix),
# computes bending stress and % diff to 'sat',
# and keeps ONLY the single best (smallest |%diff|) per wp (input RPM).

import itertools, math, csv, time, numpy
import calculations as c
import functions as fn

# target (allowable) for % diff calc
sat = 36.8403

# -------------------------
# SEARCH SPACE
# -------------------------
VARIABLES = {
    # wp: input rpm (integers) 1200..3600 inclusive step 200
    "wp":   range(1200, 3601, 200),

    # n1: stage 1 ratio (floats)
    "n1":   numpy.arange(1, 9.1, 0.1),

    # Pnd: normal diametral pitch (list of discrete choices)
    "Pnd":  [4, 5, 6, 8, 10, 12, 16, 20, 22, 25],

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

    # best result per wp: { wp: {**params, "sigma_bend":..., "percent_diff":..., "abs_percent_diff":...} }
    best_per_wp = {}

    checked = 0
    t0 = time.time()

    for values in combos:
        checked += 1
        params = dict(zip(names, values))
        wp = params["wp"]

        # your bending-stress function must accept (wp, n1, Pnd, Np1, Helix)
        sigma = c.bending_stress(**params)

        # signed % difference vs sat (your fn.distance)
        pdiff = fn.distance(sigma, sat)
        absdiff = abs(pdiff)

        # keep the single best for this wp
        if (wp not in best_per_wp) or (absdiff < best_per_wp[wp]["abs_percent_diff"]):
            best_per_wp[wp] = {
                **params,
                "sigma_bend": sigma,
                "percent_diff": pdiff,
                "abs_percent_diff": absdiff
            }

    elapsed = time.time() - t0

    # -------------------------
    # WRITE RESULTS
    # -------------------------
    with open(OUT_CSV, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(names + ["sigma_bend", "percent_diff", "abs_percent_diff"])
        for wp in sorted(best_per_wp):
            row = best_per_wp[wp]
            w.writerow([row[k] for k in names] +
                       [row["sigma_bend"], row["percent_diff"], row["abs_percent_diff"]])

    print(f"\nDone. Checked {checked:,} combinations in {elapsed:.2f}s "
          f"({checked/elapsed:,.0f} combos/s).")
    print(f"Wrote best row for each wp ({len(best_per_wp)} values) to {OUT_CSV}")

if __name__ == "__main__":
    main()
