# top5_bending_search.py
# Streams all combinations of (wp, n1, Pnd, Np1, Helix),
# calls your bending-stress function, computes %diff to allowable,
# and keeps ONLY the 5 best (smallest |%diff|).

import itertools, math, heapq, csv, time, numpy
import calculations as c
import functions as fn


sat = 36.8403  
sac = 129.242  # <-- Set your allowable contact (pitting) stress here

VARIABLES = {
    # wp: input rpm (integers)
    "wp":   range(1200, 3601, 200),

    # n1: stage 1 ratio (floats)
    "n1":   numpy.arange(1, 9.1, 0.1),

    # Pnd: normal diametral pitch (list of discrete choices)
    "Pnd":  [4, 5, 6, 8, 10],

    # Np1: pinion teeth
    "Np1":  range(10, 101, 1),

    # Helix: degrees (floats)
    "Helix": [15, 20, 25],
}

TOP_K = 5                 # keep the 5 closest to allowable
OUT_CSV = "top5_bending.csv"  # where to write the winners

# -----------------------------------------------------------
# 4) Search (streaming) — uses a max-heap trick with negatives:
#    push (-abs_pct_diff, row). If heap grows > K, pop (removes worst).
# -----------------------------------------------------------
def main():
    names = list(VARIABLES.keys())
    domains = [list(VARIABLES[k]) for k in names]
    combos = itertools.product(*domains)

    heap = []  # will store tuples (combined_metric, bending_pdiff, contact_pdiff, sigma_bend, sigma_contact, checked, params)
    checked = 0
    t0 = time.time()

    for values in combos:
        params = dict(zip(names, values))

        sigma_bend = c.bending_stress(**params)
        sigma_contact = c.contact_stress(**params)

        bend_pdiff = fn.distance(sigma_bend, sat)
        contact_pdiff = fn.distance(sigma_contact, sac)

        combined_metric = abs(bend_pdiff) + abs(contact_pdiff)
        neg_metric = -combined_metric  # Negate for max-heap

        if len(heap) < TOP_K:
            heapq.heappush(heap, (neg_metric, bend_pdiff, contact_pdiff, sigma_bend, sigma_contact, checked, params))
        else:
            heapq.heappush(heap, (neg_metric, bend_pdiff, contact_pdiff, sigma_bend, sigma_contact, checked, params))
            if len(heap) > TOP_K:
                heapq.heappop(heap)

        checked += 1
        if checked % 100000 == 0:
            elapsed = time.time() - t0
            rate = checked / max(1.0, elapsed)
            print(f"checked={checked:,}  rate={rate:,.0f}/s  best_combined_metric={-heap[0][0]:.6f}")

    # Collect winners in ascending combined metric
    winners = sorted(heap, key=lambda x: -x[0])

    # Write CSV
    with open(OUT_CSV, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            names +
            ["sigma_bend", "bending_percent_diff", "abs_bending_percent_diff",
             "sigma_contact", "contact_percent_diff", "abs_contact_percent_diff",
             "combined_metric"]
        )
        for neg_metric, bend_pdiff, contact_pdiff, sigma_bend, sigma_contact, _, params in winners:
            row = []
            for k in names:
                if k == "n1":
                    row.append(f"{params[k]:.1f}".rstrip('0').rstrip('.') if '.' in f"{params[k]:.1f}" else f"{params[k]:.1f}")
                else:
                    row.append(params[k])
            row += [
                sigma_bend, bend_pdiff, abs(bend_pdiff),
                sigma_contact, contact_pdiff, abs(contact_pdiff),
                -neg_metric  # Convert back to positive
            ]
            w.writerow(row)

    elapsed = time.time() - t0
    print(f"\nDone. Checked {checked:,} combos in {elapsed:.1f}s.")
    print(f"Wrote TOP {TOP_K} to {OUT_CSV}")

if __name__ == "__main__":
    main()
