# best_per_wp.py
# Streams all combinations of (wp, n1, Pnd, Np1, Helix),
# computes bending stress and % diff to 'sat',
# and keeps ONLY the single best (smallest |%diff|) per wp (input RPM).

import itertools, math, csv, time, numpy, os, argparse
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
    "wp":   range(1200, 3601, 10),

    # n1: stage 1 ratio (floats)
    "n1":   numpy.arange(1, 9.1, 0.1),

    # Pnd: normal diametral pitch (list of discrete choices)
    "Pnd":  [8, 10],

    # Np1: pinion teeth
    "Np1":  range(10, 101, 2),

    # Helix: degrees (floats)
    "Helix": [25],

    # Stage 2 variables (matching ranges)
    "Np2":  range(10, 101, 2),
    "Pnd2": [8, 10],
    "Helix2": [25],
}

OUT_CSV = "test.csv"

def parse_args():
    """Parse command line arguments to refine wp resolution and reuse prior results."""
    p = argparse.ArgumentParser(description="Stream best combo per wp; supports resume and wp refinement.")
    p.add_argument("--wp-start", type=int, default=1200, help="Start wp (inclusive) when using range mode.")
    p.add_argument("--wp-stop", type=int, default=3600, help="Stop wp (exclusive) when using range mode.")
    p.add_argument("--wp-step", type=int, default=10, help="Step for wp in range mode.")
    p.add_argument("--wp-values", type=str, default="", help="Comma-separated explicit wp values (overrides range mode if set).")
    p.add_argument("--out-csv", type=str, default=OUT_CSV, help="Output CSV path (defaults to data.csv).")
    return p.parse_args()

def main():
    args = parse_args()
    global OUT_CSV
    OUT_CSV = args.out_csv
    names = list(VARIABLES.keys())
    assert names[0] == "wp", "Expected 'wp' to be the first variable for per-wp streaming output"

    # Output column order: insert wf and P immediately after wp
    out_names = [names[0], 'wf', 'P'] + names[1:]

    # Build wp domain from args (explicit list overrides range)
    if args.wp_values.strip():
        wp_list = []
        for token in args.wp_values.split(','):
            token = token.strip()
            if not token:
                continue
            try:
                wp_list.append(int(token))
            except ValueError:
                raise ValueError(f"Invalid wp value '{token}' in --wp-values")
        VARIABLES['wp'] = wp_list
    else:
        VARIABLES['wp'] = range(args.wp_start, args.wp_stop, args.wp_step)

    # Prepare CSV header (write only if file doesn't exist or is empty)
    header = out_names + [
        "sigma_bend_stage1", "percent_diff_bend_stage1",
        "sigma_contact_stage1", "percent_diff_contact_stage1",
        "sigma_bend_stage2", "percent_diff_bend_stage2",
        "sigma_contact_stage2", "percent_diff_contact_stage2",
        "combined_metric"
    ]

    existing_wps = set()
    duplicate_wps = set()
    if os.path.exists(OUT_CSV) and os.path.getsize(OUT_CSV) > 0:
        # Parse existing wp values to support resume
        with open(OUT_CSV, newline="") as f:
            reader = csv.reader(f)
            existing_header = next(reader, None)
            header_ok = existing_header and len(existing_header) == len(header)
            if not header_ok:
                print("[WARN] Existing CSV header mismatch; attempting best-effort resume by parsing first column as wp anyway.")
            for row in reader:
                if not row:
                    continue
                try:
                    wp_val = int(row[0])
                    if wp_val in existing_wps:
                        duplicate_wps.add(wp_val)
                    existing_wps.add(wp_val)
                except ValueError:
                    continue
            if duplicate_wps:
                print(f"[WARN] Detected duplicate wp rows already in CSV: {sorted(list(duplicate_wps))[:10]}" + (" ..." if len(duplicate_wps) > 10 else ""))
    else:
        with open(OUT_CSV, "w", newline="") as f:
            csv.writer(f).writerow(header)

    # Helpers
    def format_row(row_dict):
        out = []
        for k in out_names:
            # format numeric parameters nicely
            if k == "n1":
                out.append(f"{row_dict[k]:.1f}".rstrip('0').rstrip('.') if '.' in f"{row_dict[k]:.1f}" else f"{row_dict[k]:.1f}")
            elif k in ("wf", "P"):
                # wf and P should be rounded to 1 decimal already; format accordingly
                val = row_dict.get(k)
                out.append(f"{val:.1f}" if isinstance(val, (int, float)) else val)
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

    def _acquire_lock(lock_path, timeout_s=30.0, poll_s=0.1):
        start = time.time()
        while True:
            try:
                fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                return fd
            except FileExistsError:
                if time.time() - start > timeout_s:
                    raise TimeoutError(f"Timeout acquiring lock {lock_path}")
                time.sleep(poll_s)

    def _release_lock(fd, lock_path):
        try:
            os.close(fd)
        finally:
            try:
                os.remove(lock_path)
            except FileNotFoundError:
                pass

    def _current_wps_in_file():
        wps = set()
        if os.path.exists(OUT_CSV) and os.path.getsize(OUT_CSV) > 0:
            with open(OUT_CSV, newline="") as f:
                reader = csv.reader(f)
                _ = next(reader, None)  # header (may not match exactly)
                for row in reader:
                    if not row:
                        continue
                    try:
                        wps.add(int(row[0]))
                    except ValueError:
                        continue
        return wps

    def append_best(row_dict):
        # Serialize append across processes and avoid duplicates under concurrency
        lock_path = OUT_CSV + ".lock"
        fd = _acquire_lock(lock_path)
        try:
            # Refresh existing wps from file to avoid duplicate appends across processes
            current = _current_wps_in_file()
            wp_val = int(row_dict["wp"]) if not isinstance(row_dict["wp"], str) else int(float(row_dict["wp"]))
            if wp_val in current:
                return False
            with open(OUT_CSV, "a", newline="") as f:
                csv.writer(f).writerow(format_row(row_dict))
            return True
        finally:
            _release_lock(fd, lock_path)

    # Pre-build domains for non-wp variables for efficiency
    rest_names = names[1:]
    rest_domains = [list(VARIABLES[k]) for k in rest_names]

    checked = 0
    written = len(existing_wps)
    t0 = time.time()

    # Iterate wp explicitly so we can skip already-completed ones without iterating their cartesian product
    for wp in VARIABLES['wp']:
        if wp in existing_wps:
            continue  # resume skip (already present in file)

        # Compute output wf and P for this wp (rounded to 1 decimal)
        wf_out = round(wp / 12.0 + 100.0, 1)
        P_out = round(wp / 240.0, 1)

        best_row_for_wp = None
        # Build combinations for remaining variables
        for rest_values in itertools.product(*rest_domains):
            params = {"wp": wp}
            params.update(dict(zip(rest_names, rest_values)))
            checked += 1

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

            params_stage2 = {
                "wp": wi,
                "n1": n2,
                "Pnd": params["Pnd2"],
                "Np1": params["Np2"],
                "Helix": params["Helix2"],
            }
            sigma_b2 = c.bending_stress(**params_stage2)
            sigma_c2 = c.contact_stress(**params_stage2)
            pdiff_b2 = fn.distance(sigma_b2, sat)
            pdiff_c2 = fn.distance(sigma_c2, sac)

            # Skip any combination where any percent diff is negative (under target)
            if (pdiff_b1 < 0) or (pdiff_c1 < 0) or (pdiff_b2 < 0) or (pdiff_c2 < 0):
                continue

            combined_metric = abs(pdiff_b1) + abs(pdiff_c1) + abs(pdiff_b2) + abs(pdiff_c2)

            if (best_row_for_wp is None) or (combined_metric < best_row_for_wp["combined_metric"]):
                best_row_for_wp = {
                    **params,
                    # include computed wf and P for output
                    "wf": wf_out,
                    "P": P_out,
                    "sigma_bend_stage1": sigma_b1,
                    "percent_diff_bend_stage1": pdiff_b1,
                    "sigma_contact_stage1": sigma_c1,
                    "percent_diff_contact_stage1": pdiff_c1,
                    "sigma_bend_stage2": sigma_b2,
                    "percent_diff_bend_stage2": pdiff_b2,
                    "sigma_contact_stage2": sigma_c2,
                    "percent_diff_contact_stage2": pdiff_c2,
                    "combined_metric": combined_metric,
                }

            # Progress print every 100k checked combos
            if checked % 100000 == 0:
                elapsed = time.time() - t0
                rate = checked / max(1.0, elapsed)
                current_metric = best_row_for_wp["combined_metric"] if best_row_for_wp else float('nan')
                print(f"checked={checked:,} rate={rate:,.0f}/s wp={wp} best_metric={current_metric:.6f} written={written}")

        # Flush this wp's best row if any valid combo found
        if best_row_for_wp is not None:
            appended = append_best(best_row_for_wp)
            if appended:
                existing_wps.add(wp)
                written += 1
            else:
                print(f"[SKIP] wp {wp} already present at append time; skipped duplicate.")

    elapsed = time.time() - t0
    # Recount rows in file for accurate total
    final_wps = _current_wps_in_file()
    # Guard against elapsed == 0 to avoid ZeroDivisionError on very fast runs
    rate = (checked / elapsed) if elapsed > 0 else 0.0
    print(f"\nDone. Checked {checked:,} combinations in {elapsed:.2f}s ({rate:,.0f} combos/s).")
    print(f"Resume-aware append complete. Total unique wp rows now in CSV: {len(final_wps)}")

if __name__ == "__main__":
    main()
