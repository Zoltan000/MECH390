"""
Local search / hill climbing for gearbox design.
Starts from multiple random seeds, then walks to better neighbors until no improvement.
For each wp, compares the best 2-stage and 3-stage candidates and stores the winner.
"""

import argparse
import csv
import os
import random
import time

from search_common import (
    CSV_HEADER,
    STAGE_KEYS,
    better,
    choice_list,
    design_row,
    evaluate_design,
    random_valid_design,
    snap_to_choice,
    sample_eval_time,
)


def neighbor_params(params, stage):
    """Generate neighboring parameter sets by stepping each variable up/down one slot."""
    for key in STAGE_KEYS[stage]:
        choices = choice_list(key)
        if len(choices) < 2:
            continue
        current_value = snap_to_choice(params[key], key)
        idx = choices.index(current_value)
        for delta in (-1, 1):
            new_idx = idx + delta
            if 0 <= new_idx < len(choices):
                new_params = params.copy()
                new_params[key] = choices[new_idx]
                yield new_params


def hill_climb_stage(wp, stage, restarts, max_steps, rng, avg_time=None, est_total=None):
    best_overall = None
    attempts = 0
    evals = 0
    t0 = time.time()
    log_every = 200
    for r in range(restarts):
        seed = random_valid_design(wp, stage, attempts=300, rng=rng)
        if not seed:
            continue
        current = seed
        for step in range(max_steps):
            attempts += 1
            improved = False
            # Evaluate all neighbors and move to the best improving one
            candidates = []
            for nparams in neighbor_params(current["params"], stage):
                res = evaluate_design(wp, nparams, stage)
                evals += 1
                if res and res["valid"]:
                    candidates.append(res)
            if candidates:
                # pick the best among improving neighbors
                best_neighbor = min(candidates, key=lambda r: r["volume"])
                if better(current, best_neighbor):
                    current = best_neighbor
                    improved = True
            if better(best_overall, current):
                best_overall = current
            if not improved:
                break
            if avg_time and est_total and evals % log_every == 0:
                elapsed = time.time() - t0
                rate = evals / elapsed if elapsed else 0
                remaining = max(est_total - evals, 0)
                eta_left = remaining * avg_time
                print(f"    Stage {stage} progress: {evals:,}/{est_total:,} evals ({rate:.1f} cps), ETA {eta_left:.1f}s")
    return best_overall, max(attempts, evals)


def load_existing(out_csv):
    """Return dict wp -> (volume, row_list)."""
    entries = {}
    if not os.path.exists(out_csv) or os.path.getsize(out_csv) == 0:
        return entries
    with open(out_csv, newline="") as f:
        reader = csv.reader(f)
        next(reader, None)
        for row in reader:
            if not row:
                continue
            try:
                wp = int(row[0])
                volume = float(row[4])
                entries[wp] = (volume, row)
            except Exception:
                continue
    return entries


def upsert_row(out_csv, header, row):
    """Insert or replace a row for the given wp if volume improves."""
    wp = int(row[0])
    volume = float(row[4])
    entries = load_existing(out_csv)
    best = entries.get(wp)
    if best and best[0] <= volume:
        return False
    entries[wp] = (volume, row)
    all_rows = [entries[k][1] for k in sorted(entries)]
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(all_rows)
    return True


def main():
    parser = argparse.ArgumentParser(description="Hill climbing search for gearbox design.")
    parser.add_argument("--wp-start", type=int, default=1200)
    parser.add_argument("--wp-stop", type=int, default=3601)
    parser.add_argument("--wp-step", type=int, default=5)
    parser.add_argument("--restarts", type=int, default=8, help="Random restarts per stage.")
    parser.add_argument("--steps", type=int, default=80, help="Max hill-climb steps per restart.")
    parser.add_argument("--out-csv", type=str, default="data_hill_climb.csv")
    parser.add_argument("--skip-3stage", action="store_true", help="Only search 2-stage designs.")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--estimate", action="store_true", help="Estimate runtime and show ETAs.")
    parser.add_argument("--sample-size", type=int, default=60, help="Samples for timing when estimating.")
    parser.add_argument("--loop", action="store_true", help="Keep rerunning wps indefinitely until interrupted.")
    args = parser.parse_args()

    rng = random.Random(args.seed)
    stage_avg = {}
    if args.estimate:
        print("Sampling evaluation time for ETA...")
        stage_avg[2] = sample_eval_time(2, samples=args.sample_size, rng=rng)
        if not args.skip_3stage:
            stage_avg[3] = sample_eval_time(3, samples=args.sample_size, rng=rng)
        for stg, avg in stage_avg.items():
            if avg:
                print(f"  Stage {stg} avg eval time: {avg:.5f}s")

    existing = load_existing(args.out_csv)
    if not existing:
        with open(args.out_csv, "w", newline="") as f:
            csv.writer(f).writerow(CSV_HEADER)

    wp_values = range(args.wp_start, args.wp_stop, args.wp_step)
    pending_wps = [wp for wp in wp_values]
    total_attempts = 0
    written = 0
    print(f"Hill climbing over {len(pending_wps)} wp values")

    def stage_eval_estimate(stage):
        neighbor_vars = sum(1 for k in STAGE_KEYS[stage] if len(choice_list(k)) > 1)
        neighbors = neighbor_vars * 2
        return args.restarts * args.steps * max(1, neighbors)

    if args.estimate and stage_avg.get(2):
        per_wp_evals = stage_eval_estimate(2)
        per_wp_time = per_wp_evals * stage_avg[2]
        if not args.skip_3stage and stage_avg.get(3):
            per_wp_evals += stage_eval_estimate(3)
            per_wp_time += stage_eval_estimate(3) * stage_avg[3]
        total_eta = per_wp_time * len(pending_wps)
        print(f"Estimated total ETA: {total_eta:.1f}s for remaining wp values")

    try:
        while True:
            for wp in pending_wps:
                print(f"\n{'='*50}\nwp = {wp}")
                est2 = stage_eval_estimate(2)
                best2, attempts2 = hill_climb_stage(
                    wp, 2, args.restarts, args.steps, rng, avg_time=stage_avg.get(2), est_total=est2
                )
                total_attempts += attempts2
                if best2:
                    print(f"  2-stage best volume: {best2['volume']:.2f}")
                else:
                    print("  2-stage: no valid design")

                best3 = None
                attempts3 = 0
                if not args.skip_3stage:
                    est3 = stage_eval_estimate(3)
                    best3, attempts3 = hill_climb_stage(
                        wp, 3, args.restarts, args.steps, rng, avg_time=stage_avg.get(3), est_total=est3
                    )
                    total_attempts += attempts3
                    if best3:
                        print(f"  3-stage best volume: {best3['volume']:.2f}")
                    else:
                        print("  3-stage: no valid design")

                winner = None
                if better(winner, best2):
                    winner = best2
                if better(winner, best3):
                    winner = best3

                if winner:
                    saved = upsert_row(args.out_csv, CSV_HEADER, design_row(winner))
                    if saved:
                        written += 1
                        print(f"  Saved {winner['stages']}-stage design (volume {winner['volume']:.2f})")
                    else:
                        print("  Existing design already better; not saved")
                else:
                    print("  No valid design for this wp")

            if not args.loop:
                break
    except KeyboardInterrupt:
        print("\nInterrupted; exiting loop.")

    print(f"\nTotal hill-climb attempts: {total_attempts:,}")
    print(f"Wrote {written} designs to {args.out_csv}")


if __name__ == "__main__":
    main()
