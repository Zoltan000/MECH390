# top5_bending_search.py
# Streams all combinations of (wp, n1, Pnd, Np1, Helix),
# calls your bending-stress function, computes %diff to allowable,
# and keeps ONLY the 5 best (smallest |%diff|).

import itertools, math, heapq, csv, time, numpy
import calculations as c
import functions as fn


sat = 36.8403  


VARIABLES = {
    # wp: input rpm (integers)
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

    heap = []  # will store tuples (-abs_pct_diff, pct_diff, sigma, params_dict)
    checked = 0
    t0 = time.time()

    for values in combos:
        params = dict(zip(names, values))

        sigma = c.bending_stress(**params)       # <-- YOUR function
        pdiff = fn.distance(sigma, sat)          # signed %
        key = -abs(pdiff)                                    # negative so min-heap pops worst when >K

        if len(heap) < TOP_K:
            heapq.heappush(heap, (key, pdiff, sigma, params))
        else:
            # push new; if worse than current K best, it will be popped immediately
            heapq.heappush(heap, (key, pdiff, sigma, params))
            if len(heap) > TOP_K:
                heapq.heappop(heap)

        checked += 1
        if checked % 100000 == 0:
            elapsed = time.time() - t0
            rate = checked / max(1.0, elapsed)
            print(f"checked={checked:,}  rate={rate:,.0f}/s  best|%diff|={abs(heap[0][0]):.6f}")

    # Collect winners in ascending |%diff|
    winners = sorted(heap, key=lambda x: abs(x[1]))

    # Write CSV
    with open(OUT_CSV, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(names + ["sigma_bend", "percent_diff", "abs_percent_diff"])
        for _, pdiff, sigma, params in winners:
            w.writerow([params[k] for k in names] + [sigma, pdiff, abs(pdiff)])

    elapsed = time.time() - t0
    print(f"\nDone. Checked {checked:,} combos in {elapsed:.1f}s.")
    print(f"Wrote TOP {TOP_K} to {OUT_CSV}")

if __name__ == "__main__":
    main()
