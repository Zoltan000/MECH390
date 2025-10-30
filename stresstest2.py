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
    # wp: input rpm (integers) 2250..2250 inclusive step 400
    "wp":   range(2250, 2251, 400),

    # n1: stage 1 ratio (floats)
    "n1":   numpy.arange(1, 9.1, 0.1),

    # Pnd: normal diametral pitch (list of discrete choices)
    "Pnd":  [4, 5, 6, 8, 10],

    # Np1: pinion teeth
    "Np1":  range(10, 101, 1),

    # Helix: degrees (floats)
    "Helix": [15, 20, 25],

    # Stage 2 variables (matching ranges)
    "Np2":  range(10, 101, 1),
    "Pnd2": [4, 5, 6, 8, 10],
    "Helix2": [15, 20, 25],
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

        # Combined metric (sum of absolute percent diffs for both stages)
        combined_metric = abs(pdiff_b1) + abs(pdiff_c1) + abs(pdiff_b2) + abs(pdiff_c2)

        # Only keep the best (lowest combined_metric) for each wp
        if (wp not in best_per_wp) or (combined_metric < best_per_wp[wp]["combined_metric"]):
            best_per_wp[wp] = {
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
            best_metric = min([v['combined_metric'] for v in best_per_wp.values()]) if best_per_wp else float('nan')
            print(f"checked={checked:,}  rate={rate:,.0f}/s  best_combined_metric={best_metric:.6f}")

    elapsed = time.time() - t0

    # -------------------------
    # WRITE RESULTS
    # -------------------------
    with open(OUT_CSV, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(names + [
            "sigma_bend_stage1", "percent_diff_bend_stage1",
            "sigma_contact_stage1", "percent_diff_contact_stage1",
            "sigma_bend_stage2", "percent_diff_bend_stage2",
            "sigma_contact_stage2", "percent_diff_contact_stage2",
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
                row["sigma_bend_stage1"], row["percent_diff_bend_stage1"],
                row["sigma_contact_stage1"], row["percent_diff_contact_stage1"],
                row["sigma_bend_stage2"], row["percent_diff_bend_stage2"],
                row["sigma_contact_stage2"], row["percent_diff_contact_stage2"],
                row["combined_metric"]
            ]
            w.writerow(output)

    print(f"\nDone. Checked {checked:,} combinations in {elapsed:.2f}s "
          f"({checked/elapsed:,.0f} combos/s).")
    print(f"Wrote best row for each wp ({len(best_per_wp)} values) to {OUT_CSV}")

if __name__ == "__main__":
    main()
