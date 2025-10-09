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
    "wp":   [2250],

    # n1: stage 1 ratio (floats)
    "n1":   numpy.arange(1, 9.1, 0.1),

    # Pnd: normal diametral pitch (list of discrete choices)
    "Pnd":  [10],

    # Np1: pinion teeth
    "Np1":  range(10, 51, 1),

    # Helix: degrees (floats)
    "Helix": [25],

    # Stage 2 variables
    "Np2":  range(10, 51, 1),
    "Pnd2": [10],
    "Helix2": [25],
}

TOP_K = 5                     # keep the 5 closest to allowable
OUT_CSV = "top5_bending2.csv"  # where to write the winners

# -----------------------------------------------------------
# Toggle this to enable/disable rejection of over-allowable stresses
REJECT_IF_OVER_ALLOWABLE = True
# 4) Search (streaming) — uses a max-heap trick with negatives:
#    push (-abs_pct_diff, row). If heap grows > K, pop (removes worst).
# -----------------------------------------------------------
def main():
    names = list(VARIABLES.keys())
    domains = [list(VARIABLES[k]) for k in names]
    combos = itertools.product(*domains)

    heap = []  # will store tuples (combined_metric, pdiff_b1, pdiff_c1, sigma_b1, sigma_c1, pdiff_b2, pdiff_c2, sigma_b2, sigma_c2, checked, params)
    checked = 0
    t0 = time.time()


    for values in combos:
        params = dict(zip(names, values))

        # Stage 1 stresses
        stage1_keys = ["wp", "n1", "Pnd", "Np1", "Helix"]
        params_stage1 = {k: params[k] for k in stage1_keys}
        sigma_b1 = c.bending_stress(**params_stage1)
        sigma_c1 = c.contact_stress(**params_stage1)
        pdiff_b1 = fn.distance(sigma_b1, sat)
        pdiff_c1 = fn.distance(sigma_c1, sac)

        # Calculate n2 and wi for stage 2
        P, Pd, wf, n, n2 = c.important_values(params["wp"], params["n1"], params["Pnd"], params["Np1"], params["Helix"])
        wi = wf

        # Stage 2 stresses
        params_stage2 = {
            "wp": wi,
            "n1": n2,
            "Pnd": params["Pnd2"],
            "Np1": params["Np2"],
            "Helix": params["Helix2"]
        }
        sigma_b2 = c.bending_stress(**params_stage2)
        sigma_c2 = c.contact_stress(**params_stage2)
        pdiff_b2 = fn.distance(sigma_b2, sat)
        pdiff_c2 = fn.distance(sigma_c2, sac)

        # Reject if any stress is above allowable
        if REJECT_IF_OVER_ALLOWABLE:
            if (
                sigma_b1 > sat or sigma_c1 > sac or
                sigma_b2 > sat or sigma_c2 > sac
            ):
                checked += 1
                continue

        # Combined metric (sum of absolute percent diffs for both stages)
        combined_metric = abs(pdiff_b1) + abs(pdiff_c1) + abs(pdiff_b2) + abs(pdiff_c2)
        neg_metric = -combined_metric  # Negate for max-heap

        if len(heap) < TOP_K:
            heapq.heappush(heap, (neg_metric, pdiff_b1, pdiff_c1, sigma_b1, sigma_c1, pdiff_b2, pdiff_c2, sigma_b2, sigma_c2, checked, params))
        else:
            heapq.heappush(heap, (neg_metric, pdiff_b1, pdiff_c1, sigma_b1, sigma_c1, pdiff_b2, pdiff_c2, sigma_b2, sigma_c2, checked, params))
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
            [
                "sigma_bend_stage1", "percent_diff_bend_stage1", "sigma_contact_stage1", "percent_diff_contact_stage1",
                "sigma_bend_stage2", "percent_diff_bend_stage2", "sigma_contact_stage2", "percent_diff_contact_stage2",
                "combined_metric"
            ]
        )
        for neg_metric, pdiff_b1, pdiff_c1, sigma_b1, sigma_c1, pdiff_b2, pdiff_c2, sigma_b2, sigma_c2, _, params in winners:
            row = []
            for k in names:
                if k == "n1":
                    row.append(f"{params[k]:.1f}".rstrip('0').rstrip('.') if '.' in f"{params[k]:.1f}" else f"{params[k]:.1f}")
                else:
                    row.append(params[k])
            row += [
                sigma_b1, pdiff_b1, sigma_c1, pdiff_c1,
                sigma_b2, pdiff_b2, sigma_c2, pdiff_c2,
                -neg_metric
            ]
            w.writerow(row)

    elapsed = time.time() - t0
    print(f"\nDone. Checked {checked:,} combos in {elapsed:.1f}s.")
    print(f"Wrote TOP {TOP_K} to {OUT_CSV}")

if __name__ == "__main__":
    main()
