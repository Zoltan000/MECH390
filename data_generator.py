# best_per_wp.py
# Streams all combinations of (wp, n1, Pnd, Np1, Helix),
# computes bending stress and % diff to 'sat',
# and keeps ONLY the single best (smallest |%diff|) per wp (input RPM).

import itertools, math, csv, time, numpy, os, random
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
    "wp":   range(1200, 3601, 100),

    # n1: stage 1 ratio (floats)
    "n1":   numpy.arange(1, 9.1, 0.5),

    # Pd1: normal diametral pitch (list of discrete choices) for stage 1
    "Pd1":  [8, 10],

    # Np1: pinion teeth
    "Np1":  range(10, 101, 5),

    # Helix: degrees (floats)
    "Helix": [15,20,25],

    # Stage 2 variables (matching ranges) - ensure Pd2 appears before Np2
    "Pd2": [8, 10],
    "Np2":  range(10, 101, 5),
    "Helix2": [15,20,25],
}

OUT_CSV = "data_sample_50k.csv"

# Simplified generator defaults: always produce a 50k random sample by default.
DEFAULT_SAMPLE_SIZE = 50000
DEFAULT_OUT_CSV = OUT_CSV

def main():
    global OUT_CSV
    # Default behavior: generate a 50k random sample CSV
    OUT_CSV = DEFAULT_OUT_CSV
    names = list(VARIABLES.keys())
    assert names[0] == "wp", "Expected 'wp' to be the first variable for per-wp streaming output"

    # Output column order: insert wf and P immediately after wp
    out_names = [names[0], 'wf', 'P'] + names[1:]

    # Use the pre-defined wp range in VARIABLES; do not allow runtime overrides.

    # Prepare CSV header (write only if file doesn't exist or is empty)
    # Columns: input variables, computed wf and P, predicted gear parameters,
    # then stage stress values, validity flag, and final gearbox volume (volume last).
    header = out_names + [
        "sigma_bend_stage1", "sigma_contact_stage1",
        "sigma_bend_stage2", "sigma_contact_stage2",
        "valid", "volume"
    ]

    # Always overwrite existing CSV so each run starts fresh
    existing_wps = set()
    with open(OUT_CSV, "w", newline="") as f:
        csv.writer(f).writerow(header)

    # Per-process part file for safe non-blocking writes when lock can't be acquired
    PART_FILE = OUT_CSV + f".part.{os.getpid()}"

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
            row_dict.get("sigma_bend_stage1", ""), row_dict.get("sigma_contact_stage1", ""),
            row_dict.get("sigma_bend_stage2", ""), row_dict.get("sigma_contact_stage2", ""),
            row_dict.get("valid", ""), row_dict.get("volume", ""),
        ]
        return out

    def _acquire_lock(lock_path, timeout_s=30.0, poll_s=0.1):
        # Try to acquire a small lock; reduce default timeout so runs started
        # from editors (which may spawn multiple processes) don't block for long.
        start = time.time()
        while True:
            try:
                fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                return fd
            except FileExistsError:
                if time.time() - start > timeout_s:
                    # Don't hold the caller indefinitely; signal failure.
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

    def _current_wps_in_part(part_path):
        wps = set()
        if os.path.exists(part_path):
            try:
                with open(part_path, newline='') as f:
                    reader = csv.reader(f)
                    for row in reader:
                        if not row:
                            continue
                        try:
                            wps.add(int(row[0]))
                        except ValueError:
                            continue
            except Exception:
                pass
        return wps

    def append_best(row_dict):
        # Write best rows to the per-process PART_FILE to avoid repeatedly
        # opening the main CSV during the run (prevents blocking in editors).
        try:
            current = _current_wps_in_file().union(_current_wps_in_part(PART_FILE))
            wp_val = int(row_dict["wp"]) if not isinstance(row_dict["wp"], str) else int(float(row_dict["wp"]))
            if wp_val in current:
                return False
            with open(PART_FILE, "a", newline="") as f:
                csv.writer(f).writerow(format_row(row_dict))
            return True
        except Exception:
            return False

    def append_any(row_dict):
        """Append a row without checking for duplicate wp (used in 'all' and 'sample' modes)."""
        # Always append to per-process PART_FILE to avoid blocking on main CSV
        try:
            with open(PART_FILE, "a", newline="") as f:
                csv.writer(f).writerow(format_row(row_dict))
            return True
        except Exception:
            return False

    # Pre-build domains for non-wp variables for efficiency
    rest_names = names[1:]
    rest_domains = [list(VARIABLES[k]) for k in rest_names]

    checked = 0
    written = len(existing_wps)
    t0 = time.time()

    # Force sample mode for the simplified generator
    mode = 'sample'

    # Iterate according to mode
    if mode == "best":
        # Iterate wp explicitly so we can skip already-completed ones without iterating their cartesian product
        for wp in VARIABLES['wp']:
            if wp in existing_wps:
                continue  # resume skip (already present in file)

            best_row_for_wp = None
            for rest_values in itertools.product(*rest_domains):
                params = {"wp": wp}
                params.update(dict(zip(rest_names, rest_values)))
                checked += 1

                try:
                    res = c.results(params["wp"], params["n1"], None,
                                    params["Pd1"], params["Np1"], params["Helix"],
                                    params["Pd2"], params["Np2"], params["Helix2"],
                                    None, None, None)
                except Exception:
                    continue

                wf_calc, P_calc, Volume_calc, sigma_b1, sigma_c1, sigma_b2, sigma_c2 = res

                pdiff_b1 = fn.distance(sigma_b1, sat)
                pdiff_c1 = fn.distance(sigma_c1, sac)
                pdiff_b2 = fn.distance(sigma_b2, sat)
                pdiff_c2 = fn.distance(sigma_c2, sac)

                combined_metric = abs(pdiff_b1) + abs(pdiff_c1) + abs(pdiff_b2) + abs(pdiff_c2)

                if (best_row_for_wp is None) or (combined_metric < best_row_for_wp["combined_metric"]):
                    best_row_for_wp = {
                        **params,
                        "wf": round(float(wf_calc), 1),
                        "P": round(float(P_calc), 1),
                        "sigma_bend_stage1": float(sigma_b1),
                        "sigma_contact_stage1": float(sigma_c1),
                        "sigma_bend_stage2": float(sigma_b2),
                        "sigma_contact_stage2": float(sigma_c2),
                        "pdiff_b1": float(pdiff_b1),
                        "pdiff_c1": float(pdiff_c1),
                        "pdiff_b2": float(pdiff_b2),
                        "pdiff_c2": float(pdiff_c2),
                        "valid": all([sigma_b1 < sat, sigma_c1 < sac, sigma_b2 < sat, sigma_c2 < sac]),
                        "volume": float(Volume_calc),
                        "combined_metric": combined_metric,
                    }

                if checked % 100000 == 0:
                    elapsed = time.time() - t0
                    rate = checked / max(1.0, elapsed)
                    current_metric = best_row_for_wp["combined_metric"] if best_row_for_wp else float('nan')
                    print(f"checked={checked:,} rate={rate:,.0f}/s wp={wp} best_metric={current_metric:.6f} written={written}")

            if best_row_for_wp is not None:
                appended = append_best(best_row_for_wp)
                if appended:
                    existing_wps.add(wp)
                    written += 1
                else:
                    print(f"[SKIP] wp {wp} already present at append time; skipped duplicate.")

    elif mode == "all":
        # Write every combination (no skipping). This can produce many rows.
        for wp in VARIABLES['wp']:
            for rest_values in itertools.product(*rest_domains):
                params = {"wp": wp}
                params.update(dict(zip(rest_names, rest_values)))
                checked += 1
                try:
                    res = c.results(params["wp"], params["n1"], None,
                                    params["Pd1"], params["Np1"], params["Helix"],
                                    params["Pd2"], params["Np2"], params["Helix2"],
                                    None, None, None)
                except Exception:
                    continue

                wf_calc, P_calc, Volume_calc, sigma_b1, sigma_c1, sigma_b2, sigma_c2 = res

                pdiff_b1 = fn.distance(sigma_b1, sat)
                pdiff_c1 = fn.distance(sigma_c1, sac)
                pdiff_b2 = fn.distance(sigma_b2, sat)
                pdiff_c2 = fn.distance(sigma_c2, sac)

                row = {
                    **params,
                    "wf": round(float(wf_calc), 1),
                    "P": round(float(P_calc), 1),
                    "sigma_bend_stage1": float(sigma_b1),
                    "sigma_contact_stage1": float(sigma_c1),
                    "sigma_bend_stage2": float(sigma_b2),
                    "sigma_contact_stage2": float(sigma_c2),
                    "pdiff_b1": float(pdiff_b1),
                    "pdiff_c1": float(pdiff_c1),
                    "pdiff_b2": float(pdiff_b2),
                    "pdiff_c2": float(pdiff_c2),
                    "valid": all([sigma_b1 < sat, sigma_c1 < sac, sigma_b2 < sat, sigma_c2 < sac]),
                    "volume": float(Volume_calc),
                }
                append_any(row)
                written += 1

                if checked % 100000 == 0:
                    elapsed = time.time() - t0
                    rate = checked / max(1.0, elapsed)
                    print(f"checked={checked:,} rate={rate:,.0f}/s written={written}")

    elif mode == "sample":
        sample_size = DEFAULT_SAMPLE_SIZE
        attempts = 0
        max_attempts = sample_size * 10
        while written < sample_size and attempts < max_attempts:
            attempts += 1
            wp = random.choice(list(VARIABLES['wp']))
            rest_values = [random.choice(domain) for domain in rest_domains]
            params = {"wp": wp}
            params.update(dict(zip(rest_names, rest_values)))
            checked += 1
            try:
                res = c.results(params["wp"], params["n1"], None,
                                params["Pd1"], params["Np1"], params["Helix"],
                                params["Pd2"], params["Np2"], params["Helix2"],
                                None, None, None)
            except Exception:
                continue

            wf_calc, P_calc, Volume_calc, sigma_b1, sigma_c1, sigma_b2, sigma_c2 = res

            pdiff_b1 = fn.distance(sigma_b1, sat)
            pdiff_c1 = fn.distance(sigma_c1, sac)
            pdiff_b2 = fn.distance(sigma_b2, sat)
            pdiff_c2 = fn.distance(sigma_c2, sac)

            row = {
                **params,
                "wf": round(float(wf_calc), 1),
                "P": round(float(P_calc), 1),
                "sigma_bend_stage1": float(sigma_b1),
                "sigma_contact_stage1": float(sigma_c1),
                "sigma_bend_stage2": float(sigma_b2),
                "sigma_contact_stage2": float(sigma_c2),
                "pdiff_b1": float(pdiff_b1),
                "pdiff_c1": float(pdiff_c1),
                "pdiff_b2": float(pdiff_b2),
                "pdiff_c2": float(pdiff_c2),
                "valid": all([sigma_b1 < sat, sigma_c1 < sac, sigma_b2 < sat, sigma_c2 < sac]),
                "volume": float(Volume_calc),
            }
            append_any(row)
            written += 1

            if written % 1000 == 0:
                elapsed = time.time() - t0
                rate = checked / max(1.0, elapsed)
                print(f"sampled={written:,} attempts={attempts:,} checked={checked:,} rate={rate:,.0f}/s")

    elapsed = time.time() - t0
    # Recount rows in file for accurate total
    final_wps = _current_wps_in_file()
    # Guard against elapsed == 0 to avoid ZeroDivisionError on very fast runs
    rate = (checked / elapsed) if elapsed > 0 else 0.0
    # Attempt to merge any per-process PART_FILE into the main CSV under the lock
    try:
        if os.path.exists(PART_FILE):
            lock_path = OUT_CSV + ".lock"
            try:
                fd = _acquire_lock(lock_path, timeout_s=5.0)
            except TimeoutError:
                print(f"Warning: couldn't acquire lock to merge part file {PART_FILE}; left as-is.")
            else:
                try:
                    # Append part contents to main CSV
                    with open(OUT_CSV, "a", newline="") as fout, open(PART_FILE, newline="") as fin:
                        for line in fin:
                            fout.write(line)
                    os.remove(PART_FILE)
                    # refresh final count after merge
                    final_wps = _current_wps_in_file()
                finally:
                    _release_lock(fd, lock_path)

    except Exception as e:
        print(f"Non-fatal error while merging part file: {e}")

    print(f"\nDone. Checked {checked:,} combinations in {elapsed:.2f}s ({rate:,.0f} combos/s).")
    print(f"Resume-aware append complete. Total unique wp rows now in CSV: {len(final_wps)}")

if __name__ == "__main__":
    main()
