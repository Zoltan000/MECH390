# best_per_wp.py
# Streams all combinations of (wp, n1, Pnd, Np1, Helix),
# computes bending stress and % diff to 'sat',
# and keeps ONLY the single best (smallest |%diff|) per wp (input RPM).

import itertools, math, csv, time, numpy, os
import calculations as c
import functions as fn

# target (allowable) for % diff calc
sat = 36.8403
sac = 129.242

# -------------------------
# SEARCH SPACE
# -------------------------
VARIABLES = {
    # wp: input rpm (integers) 1200..33600 inclusive step 100
    "wp":   range(1200, 1201, 100),

    # n1: stage 1 ratio (floats)
    "n1":   numpy.arange(1, 9.1, 0.1),

    # Pnd: normal diametral pitch (list of discrete choices)
    "Pnd":  [4, 5, 6, 8, 10],

    # Np1: pinion teeth
    "Np1":  range(10, 101, 2),

    # Helix: degrees (floats)
    "Helix": [15, 20, 25],

    # Stage 2 variables (matching ranges)
    "Np2":  range(10, 101, 2),
    "Pnd2": [4, 5, 6, 8, 10],
    "Helix2": [15, 20, 25],
}

OUT_CSV = "data_copy.csv"

def main():
    names   = list(VARIABLES.keys())
    domains = [list(VARIABLES[k]) for k in names]
    combos  = itertools.product(*domains)

    # Ensure 'wp' is the slowest-varying dimension so we can flush per-wp
    assert names[0] == "wp", "Expected 'wp' to be the first variable for per-wp streaming output"

    # Prepare CSV header (write only if file doesn't exist or is empty)
    header = names + [
        "sigma_bend_stage1", "percent_diff_bend_stage1",
        "sigma_contact_stage1", "percent_diff_contact_stage1",
        "sigma_bend_stage2", "percent_diff_bend_stage2",
        "sigma_contact_stage2", "percent_diff_contact_stage2",
        "combined_metric"
    ]
    if (not os.path.exists(OUT_CSV)) or (os.path.getsize(OUT_CSV) == 0):
        with open(OUT_CSV, "w", newline="") as f:
            csv.writer(f).writerow(header)

    # Helpers
    def format_row(row_dict):
        out = []
        for k in names:
            if k == "n1":
                out.append(f"{row_dict[k]:.1f}".rstrip('0').rstrip('.') if '.' in f"{row_dict[k]:.1f}" else f"{row_dict[k]:.1f}")
            else:
                out.append(row_dict[k])
        out += [
            row_dict["sigma_bend_stage1"], row_dict["percent_diff_bend_stage1"],
            row_dict["sigma_contact_stage1"], row_dict["percent_diff_contact_stage1"],
            row_dict["sigma_bend_stage2"], row_dict["percent_diff_bend_stage2"],
            row_dict["sigma_contact_stage2"], row_dict["percent_diff_contact_stage2"],
            row_dict["combined_metric"],
        ]
        return out

    def append_best(row_dict):
        with open(OUT_CSV, "a", newline="") as f:
            csv.writer(f).writerow(format_row(row_dict))

    checked = 0
    written = 0
    t0 = time.time()

    current_wp = None
    best_row_for_wp = None

    for values in combos:
        checked += 1
        params = dict(zip(names, values))
        wp = params["wp"]

        # Flush when wp changes (meaning we finished all combos for the previous wp)
        if current_wp is None:
            current_wp = wp
            best_row_for_wp = None
        elif wp != current_wp:
            if best_row_for_wp is not None:
                append_best(best_row_for_wp)
                written += 1
            current_wp = wp
            best_row_for_wp = None

        # Stage 1 stresses (only pass expected args)
        stage1_keys = ["wp", "n1", "Pnd", "Np1", "Helix"]
        params_stage1 = {k: params[k] for k in stage1_keys}
        sigma_b1 = c.bending_stress(**params_stage1)
        sigma_c1 = c.contact_stress(**params_stage1)
        pdiff_b1 = fn.distance(sigma_b1, sat)
        pdiff_c1 = fn.distance(sigma_c1, sac)

        # Calculate n2 and wi for stage 2
        P, Pd, wf, n, n2 = c.important_values(params["wp"], params["n1"], params["Pnd"], params["Np1"], params["Helix"])
        wi = wf

        # Stage 2 stresses (only pass expected args, using stage 2 variables)
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

        # Skip any combination where any percent diff is negative (under target)
        if (pdiff_b1 < 0) or (pdiff_c1 < 0) or (pdiff_b2 < 0) or (pdiff_c2 < 0):
            continue

        # Combined metric (sum of absolute percent diffs for both stages)
        combined_metric = abs(pdiff_b1) + abs(pdiff_c1) + abs(pdiff_b2) + abs(pdiff_c2)

        # Keep the single best (lowest combined_metric) for the current wp
        if (best_row_for_wp is None) or (combined_metric < best_row_for_wp["combined_metric"]):
            best_row_for_wp = {
                **params,
                "sigma_bend_stage1": sigma_b1,
                "percent_diff_bend_stage1": pdiff_b1,
                "sigma_contact_stage1": sigma_c1,
                "percent_diff_contact_stage1": pdiff_c1,
                "sigma_bend_stage2": sigma_b2,
                "percent_diff_bend_stage2": pdiff_b2,
                "sigma_contact_stage2": sigma_c2,
                "percent_diff_contact_stage2": pdiff_c2,
                "combined_metric": combined_metric
            }

        # Periodic progress print
        if checked % 100000 == 0:
            elapsed = time.time() - t0
            rate = checked / max(1.0, elapsed)
            best_metric = best_row_for_wp["combined_metric"] if best_row_for_wp else float('nan')
            print(f"checked={checked:,}  rate={rate:,.0f}/s  current_wp={current_wp}  best_combined_metric={best_metric:.6f}  written={written}")

    # Flush the final wp
    if best_row_for_wp is not None:
        append_best(best_row_for_wp)
        written += 1

    elapsed = time.time() - t0

    print(f"\nDone. Checked {checked:,} combinations in {elapsed:.2f}s "
          f"({checked/elapsed:,.0f} combos/s).")
    print(f"Appended best row for each wp (total {written}) to {OUT_CSV}")

if __name__ == "__main__":
    main()
