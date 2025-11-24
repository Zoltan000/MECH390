# data_generator.py
# For each wp (input RPM), finds the minimum volume gearbox design
# that satisfies all stress constraints, comparing both 2-stage and 3-stage configurations.

import itertools, csv, time, numpy, os, argparse, random
from concurrent.futures import ProcessPoolExecutor
import calculations as c

# Maximum allowable stresses
MAX_BENDING_STRESS = 36.8403  # ksi
MAX_CONTACT_STRESS = 129.242  # ksi

# -------------------------
# SEARCH SPACE
# -------------------------
VARIABLES = {
    # wp: input rpm (integers)
    "wp":  range(1200, 3601, 100),

    # n1: stage 1 ratio (floats)
    "n1":   numpy.arange(1, 9.1, 0.1),

    # Pd: diametral pitch (list of discrete choices)
    "Pd1":  [8, 10],

    # Np1: pinion teeth
    "Np1":  range(10, 101, 2),

    # Helix1: degrees
    "Helix1": [20, 25],

    # Stage 2 variables
    "Np2":  range(10, 101, 2),
    "Pd2": [6, 8, 10],
    "Helix2": [15, 20, 25],

    # Stage 3 variables (for 3-stage gearboxes)
    "n2":   numpy.arange(1, 9.1, 0.1),
    "Np3":  range(10, 101, 1),
    "Pd3": [4, 5, 6, 8, 10],
    "Helix3": [15, 20, 25],
}

VALUE_LISTS = {k: list(v) for k, v in VARIABLES.items()}
TWO_STAGE_KEYS = ("n1", "Pd1", "Np1", "Helix1", "Pd2", "Np2", "Helix2")
THREE_STAGE_KEYS = ("n1", "Pd1", "Np1", "Helix1", "n2", "Pd2", "Np2", "Helix2", "Pd3", "Np3", "Helix3")

OUT_CSV = "TEST.csv"

def parse_args():
    """Parse command line arguments."""
    p = argparse.ArgumentParser(description="Find minimum volume gearbox designs for each wp.")
    p.add_argument("--wp-start", type=int, default=1200, help="Start wp (inclusive).")
    p.add_argument("--wp-stop", type=int, default=3601, help="Stop wp (exclusive).")
    p.add_argument("--wp-step", type=int, default=10, help="Step for wp.")
    p.add_argument("--out-csv", type=str, default=OUT_CSV, help="Output CSV path.")
    p.add_argument("--estimate", action="store_true", help="Estimate total combinations and runtime, then exit.")
    p.add_argument("--sample-size", type=int, default=100000, help="Number of sample combinations to time when estimating.")
    p.add_argument("--only-2stage", action="store_true", help="Run only 2-stage search (skip 3-stage). Temporary convenience flag.", default=True)
    p.add_argument("--num-workers", type=int, default=os.cpu_count() or 1, help="Parallel workers (processes).")
    p.add_argument("--samples-2stage", type=int, default=50000, help="Max combinations to evaluate per-wp for 2-stage. 0 = exhaustive (slow).")
    p.add_argument("--samples-3stage", type=int, default=50000, help="Max combinations to evaluate per-wp for 3-stage. 0 = exhaustive (slow).")
    p.add_argument("--seed", type=int, default=1234, help="PRNG seed for sampling.")
    return p.parse_args()

def is_valid_design(stresses):
    """Check if all stresses are below maximum allowable values."""
    return all(stress <= max_stress for stress, max_stress in [
        (stresses[0], MAX_BENDING_STRESS),  # Stage 1 bending
        (stresses[1], MAX_CONTACT_STRESS),  # Stage 1 contact
        (stresses[2], MAX_BENDING_STRESS),  # Stage 2 bending
        (stresses[3], MAX_CONTACT_STRESS),  # Stage 2 contact
    ] if stress is not None)


def _combo_stream(keys, max_samples, rng):
    """Yield combinations from deterministic sampling or exhaustive product."""
    if max_samples and max_samples > 0:
        for _ in range(max_samples):
            yield tuple(rng.choice(VALUE_LISTS[k]) for k in keys)
        return

    yield from itertools.product(*(VALUE_LISTS[k] for k in keys))


def _eval_2stage(args):
    wp, combo = args
    n1, Pd1, Np1, Helix1, Pd2, Np2, Helix2 = combo
    try:
        wf, P, volume, s1b, s1c, s2b, s2c, _, _ = c.results(
            wp, n1, None, Pd1, Np1, Helix1, Pd2, Np2, Helix2, None, None, None
        )
    except Exception:
        return None

    if not is_valid_design([s1b, s1c, s2b, s2c]):
        return None

    return volume, {
        'wp': wp, 'wf': wf, 'P': P,
        'n1': n1, 'Pd1': Pd1, 'Np1': Np1, 'Helix1': Helix1,
        'Pd2': Pd2, 'Np2': Np2, 'Helix2': Helix2,
        'n2': None, 'Pd3': None, 'Np3': None, 'Helix3': None,
        'volume': volume,
        's1_bending': s1b, 's1_contact': s1c,
        's2_bending': s2b, 's2_contact': s2c,
        's3_bending': None, 's3_contact': None,
        'stages': 2
    }


def _eval_3stage(args):
    wp, combo = args
    n1, Pd1, Np1, Helix1, n2, Pd2, Np2, Helix2, Pd3, Np3, Helix3 = combo
    try:
        wf, P, volume, s1b, s1c, s2b, s2c, s3b, s3c = c.results(
            wp, n1, n2, Pd1, Np1, Helix1, Pd2, Np2, Helix2, Pd3, Np3, Helix3
        )
    except Exception:
        return None

    if not is_valid_design([s1b, s1c, s2b, s2c]):
        return None
    if s3b is None or s3c is None or s3b > MAX_BENDING_STRESS or s3c > MAX_CONTACT_STRESS:
        return None

    return volume, {
        'wp': wp, 'wf': wf, 'P': P,
        'n1': n1, 'Pd1': Pd1, 'Np1': Np1, 'Helix1': Helix1,
        'n2': n2, 'Pd2': Pd2, 'Np2': Np2, 'Helix2': Helix2,
        'Pd3': Pd3, 'Np3': Np3, 'Helix3': Helix3,
        'volume': volume,
        's1_bending': s1b, 's1_contact': s1c,
        's2_bending': s2b, 's2_contact': s2c,
        's3_bending': s3b, 's3_contact': s3c,
        'stages': 3
    }


def _parallel_search(wp, combos, eval_fn, workers):
    """Evaluate combos (optionally in parallel), keeping minimum feasible volume."""
    best = None
    best_volume = float('inf')
    checked = 0

    def _sequential():
        nonlocal checked, best_volume, best
        for combo in combos:
            checked += 1
            res = eval_fn((wp, combo))
            if res and res[0] < best_volume:
                best_volume, best = res
        return best, checked

    if workers == 1:
        return _sequential()

    try:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            for res in ex.map(eval_fn, ((wp, combo) for combo in combos), chunksize=64):
                checked += 1
                if res and res[0] < best_volume:
                    best_volume, best = res
        return best, checked
    except Exception as exc:
        print(f"    Parallel execution failed for wp={wp} ({exc}); falling back to single-process.")
        return _sequential()


def find_best_2stage(wp, max_samples, workers, rng):
    combos = _combo_stream(TWO_STAGE_KEYS, max_samples, rng)
    return _parallel_search(wp, combos, _eval_2stage, workers)


def find_best_3stage(wp, max_samples, workers, rng):
    combos = _combo_stream(THREE_STAGE_KEYS, max_samples, rng)
    return _parallel_search(wp, combos, _eval_3stage, workers)


def main():
    args = parse_args()
    global OUT_CSV
    OUT_CSV = args.out_csv
    
    # Build wp range
    VARIABLES['wp'] = range(args.wp_start, args.wp_stop, args.wp_step)
    
    # CSV header
    header = [
        'wp', 'wf', 'P', 'stages', 'volume',
        'n1', 'Pd1', 'Np1', 'Helix1',
        'n2', 'Pd2', 'Np2', 'Helix2',
        'Pd3', 'Np3', 'Helix3',
        's1_bending', 's1_contact',
        's2_bending', 's2_contact',
        's3_bending', 's3_contact'
    ]
    
    # Check existing wp values to support resume
    existing_wps = set()
    if os.path.exists(OUT_CSV) and os.path.getsize(OUT_CSV) > 0:
        with open(OUT_CSV, newline="") as f:
            reader = csv.reader(f)
            next(reader, None)  # skip header
            for row in reader:
                if row:
                    try:
                        existing_wps.add(int(row[0]))
                    except (ValueError, IndexError):
                        continue
    else:
        # Write header if file doesn't exist
        with open(OUT_CSV, "w", newline="") as f:
            csv.writer(f).writerow(header)

    # If user only wants an estimate, compute counts and a rough runtime estimate
    if args.estimate:
        def count_items(seq):
            try:
                return len(seq)
            except Exception:
                return sum(1 for _ in seq)

        n1_len = count_items(VARIABLES['n1'])
        Pd1_len = count_items(VARIABLES['Pd1'])
        Np1_len = count_items(VARIABLES['Np1'])
        Helix1_len = count_items(VARIABLES['Helix1'])
        Pd2_len = count_items(VARIABLES['Pd2'])
        Np2_len = count_items(VARIABLES['Np2'])
        Helix2_len = count_items(VARIABLES['Helix2'])
        n2_len = count_items(VARIABLES['n2'])
        Pd3_len = count_items(VARIABLES['Pd3'])
        Np3_len = count_items(VARIABLES['Np3'])
        Helix3_len = count_items(VARIABLES['Helix3'])

        two_stage_per_wp = (n1_len * Pd1_len * Np1_len * Helix1_len * Pd2_len * Np2_len * Helix2_len)
        three_stage_per_wp = (n1_len * Pd1_len * Np1_len * Helix1_len * n2_len * Pd2_len * Np2_len * Helix2_len * Pd3_len * Np3_len * Helix3_len)

        if args.only_2stage:
            three_stage_per_wp = 0

        try:
            wp_count = count_items(VARIABLES['wp'])
        except Exception:
            wp_count = sum(1 for _ in VARIABLES['wp'])

        per_wp = two_stage_per_wp + three_stage_per_wp
        total_all = per_wp * wp_count
        total_remaining = per_wp * max(0, wp_count - len(existing_wps))

        print(f"Estimate mode: per-wp combinations -> 2-stage={two_stage_per_wp:,}, 3-stage={three_stage_per_wp:,}, total={per_wp:,}")
        print(f"Total wp values: {wp_count:,} (remaining: {max(0, wp_count - len(existing_wps)):,})")
        print(f"Total combinations (all wp): {total_all:,}")
        print(f"Total combinations (remaining): {total_remaining:,}")

        # Sample a few random combinations to estimate time per combination
        samples = max(1, min(args.sample_size, 100000))
        sample_attempts = 0
        sample_time = 0.0
        sample_wp_choices = list(VARIABLES['wp'])

        # Limit max samples if combinations are fewer
        max_possible = per_wp if per_wp > 0 else 1
        samples = min(samples, max_possible)

        print(f"Timing {samples} sample combinations to estimate runtime (this may call calculation routines)...")

        for i in range(samples):
            # pick a random wp and random combination
            wp_sample = random.choice(sample_wp_choices)

            # choose stage proportional to counts
            if per_wp == 0:
                pick_3 = False
            else:
                pick_3 = random.random() < (three_stage_per_wp / per_wp)

            if pick_3:
                # random 3-stage combination
                n1 = random.choice(list(VARIABLES['n1']))
                n2 = random.choice(list(VARIABLES['n2']))
                Pd1 = random.choice(list(VARIABLES['Pd1']))
                Np1 = random.choice(list(VARIABLES['Np1']))
                Helix1 = random.choice(list(VARIABLES['Helix1']))
                Pd2 = random.choice(list(VARIABLES['Pd2']))
                Np2 = random.choice(list(VARIABLES['Np2']))
                Helix2 = random.choice(list(VARIABLES['Helix2']))
                Pd3 = random.choice(list(VARIABLES['Pd3']))
                Np3 = random.choice(list(VARIABLES['Np3']))
                Helix3 = random.choice(list(VARIABLES['Helix3']))

                start = time.time()
                try:
                    # call results -- exceptions are expected for some combinations
                    _ = c.results(wp_sample, n1, n2, Pd1, Np1, Helix1, Pd2, Np2, Helix2, Pd3, Np3, Helix3)
                except Exception:
                    pass
                sample_time += (time.time() - start)
                sample_attempts += 1
            else:
                # random 2-stage combination
                n1 = random.choice(list(VARIABLES['n1']))
                Pd1 = random.choice(list(VARIABLES['Pd1']))
                Np1 = random.choice(list(VARIABLES['Np1']))
                Helix1 = random.choice(list(VARIABLES['Helix1']))
                Pd2 = random.choice(list(VARIABLES['Pd2']))
                Np2 = random.choice(list(VARIABLES['Np2']))
                Helix2 = random.choice(list(VARIABLES['Helix2']))

                start = time.time()
                try:
                    _ = c.results(wp_sample, n1, None, Pd1, Np1, Helix1, Pd2, Np2, Helix2, None, None, None)
                except Exception:
                    pass
                sample_time += (time.time() - start)
                sample_attempts += 1

        if sample_attempts == 0 or sample_time == 0.0:
            print("Unable to measure sample timings; estimation not available.")
            return

        avg_time = sample_time / sample_attempts

        est_seconds = avg_time * total_remaining

        def fmt_secs(s):
            if s < 60:
                return f"{s:.1f}s"
            m = s / 60.0
            if m < 60:
                return f"{m:.1f}m ({s:.0f}s)"
            h = m / 60.0
            if h < 24:
                return f"{h:.2f}h ({m:.0f}m)"
            d = h / 24.0
            if d < 365:
                return f"{d:.2f}d ({h:.0f}h)"
            y= d / 365.0
            return f"{y:.2f}y ({d:.0f}d)"

        print(f"Sampled {sample_attempts} calls, total sample time: {sample_time:.2f}s, avg per-call: {avg_time:.4f}s")
        print(f"Estimated runtime for remaining {total_remaining:,} combinations: {fmt_secs(est_seconds)}")
        return
    
    print(f"Starting optimization for wp values: {args.wp_start} to {args.wp_stop-1} step {args.wp_step}")
    print(f"Output file: {OUT_CSV}")
    print(f"Resuming: skipping {len(existing_wps)} already completed wp values")
    
    t_start = time.time()
    total_checked = 0
    written = 0
    
    for wp in VARIABLES['wp']:
        if wp in existing_wps:
            continue
        
        wp_rng = random.Random(args.seed + wp)
        wp_start = time.time()
        print(f"\n{'='*60}")
        print(f"Processing wp = {wp} RPM")
        
        # Find best 2-stage design
        print("  Searching 2-stage designs...")
        best_2stage, checked_2 = find_best_2stage(wp, args.samples_2stage, args.num_workers, wp_rng)
        total_checked += checked_2
        print(f"    Checked {checked_2:,} combinations")
        if best_2stage:
            print(f"    Best 2-stage volume: {best_2stage['volume']:.2f} in³")
        else:
            print(f"    No valid 2-stage design found")
        
        # Find best 3-stage design (allow skipping for 2-stage-only runs)
        if not args.only_2stage:
            print("  Searching 3-stage designs...")
            best_3stage, checked_3 = find_best_3stage(wp, args.samples_3stage, args.num_workers, wp_rng)
            total_checked += checked_3
            print(f"    Checked {checked_3:,} combinations")
            if best_3stage:
                print(f"    Best 3-stage volume: {best_3stage['volume']:.2f} in³")
            else:
                print(f"    No valid 3-stage design found")
        else:
            best_3stage = None
            checked_3 = 0
            print("  Skipping 3-stage search (only-2stage set)")
        
        # Compare and select the best overall design
        best_design = None
        if best_2stage and best_3stage:
            best_design = best_2stage if best_2stage['volume'] < best_3stage['volume'] else best_3stage
            print(f"  Winner: {best_design['stages']}-stage design (volume: {best_design['volume']:.2f} in³)")
        elif best_2stage:
            best_design = best_2stage
            print(f"  Winner: 2-stage design (only valid option)")
        elif best_3stage:
            best_design = best_3stage
            print(f"  Winner: 3-stage design (only valid option)")
        else:
            print(f"  No valid design found for wp={wp}")
        
        # Write to CSV
        if best_design:
            row = [
                best_design['wp'], best_design['wf'], best_design['P'],
                best_design['stages'], best_design['volume'],
                best_design['n1'], best_design['Pd1'], best_design['Np1'], best_design['Helix1'],
                best_design.get('n2', ''), best_design['Pd2'], best_design['Np2'], best_design['Helix2'],
                best_design.get('Pd3', ''), best_design.get('Np3', ''), best_design.get('Helix3', ''),
                best_design['s1_bending'], best_design['s1_contact'],
                best_design['s2_bending'], best_design['s2_contact'],
                best_design.get('s3_bending', ''), best_design.get('s3_contact', '')
            ]
            with open(OUT_CSV, "a", newline="") as f:
                csv.writer(f).writerow(row)
            written += 1
        
        wp_elapsed = time.time() - wp_start
        print(f"  Time for this wp: {wp_elapsed:.1f}s")
    
    elapsed = time.time() - t_start
    print(f"\n{'='*60}")
    print(f"Optimization complete!")
    print(f"Total combinations checked: {total_checked:,}")
    print(f"Designs written: {written}")
    print(f"Total time: {elapsed:.1f}s ({elapsed/60:.1f} min)")
    if written > 0:
        print(f"Average time per wp: {elapsed/written:.1f}s")

if __name__ == "__main__":
    main()
