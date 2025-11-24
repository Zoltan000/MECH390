"""
Evolutionary / genetic search for gearbox design.
Maintains a small population, selects via tournament, and mutates genes.
For each wp, searches both 2- and 3-stage designs and stores the better option.
"""

import argparse
import csv
import os
import random
import time

from search_common import (
    CSV_HEADER,
    MAX_BENDING_STRESS,
    MAX_CONTACT_STRESS,
    STAGE_KEYS,
    better,
    choice_list,
    design_row,
    evaluate_design,
    random_params,
    sample_eval_time,
)


def stress_penalty(stresses):
    """Compute a soft penalty based on how far stresses exceed limits."""
    penalty = 0.0
    checks = [
        ("s1_bending", MAX_BENDING_STRESS),
        ("s1_contact", MAX_CONTACT_STRESS),
        ("s2_bending", MAX_BENDING_STRESS),
        ("s2_contact", MAX_CONTACT_STRESS),
        ("s3_bending", MAX_BENDING_STRESS),
        ("s3_contact", MAX_CONTACT_STRESS),
    ]
    for key, limit in checks:
        val = stresses.get(key)
        if val is None:
            penalty += 1.0
        elif val > limit:
            penalty += (val - limit) / max(limit, 1e-6)
    return penalty


def fitness(res):
    """Lower is better. Invalid designs get a big penalty."""
    if res is None:
        return 1e12
    if res["valid"]:
        return res["volume"]
    return 1e9 + stress_penalty(res["stresses"]) * 1e7 + res["volume"]


def tournament(evaluated, k, rng):
    sample = rng.sample(evaluated, k)
    return min(sample, key=lambda item: item[2])[0]


def crossover(p1, p2, stage, rng):
    child = {}
    for key in STAGE_KEYS[stage]:
        child[key] = p1[key] if rng.random() < 0.5 else p2[key]
    return child


def mutate(params, stage, rate, rng):
    mutated = params.copy()
    for key in STAGE_KEYS[stage]:
        if rng.random() < rate:
            mutated[key] = rng.choice(choice_list(key))
    return mutated


def genetic_stage(wp, stage, pop_size, generations, mutation_rate, elite_frac, rng, avg_time=None):
    population = [random_params(stage, rng) for _ in range(pop_size)]
    best_valid = None
    evals = 0
    elite_count = max(1, int(pop_size * elite_frac))
    tournament_k = max(2, min(5, pop_size // 2))
    est_total = pop_size * generations
    t0 = time.time()

    for _ in range(generations):
        evaluated = []
        for p in population:
            res = evaluate_design(wp, p, stage)
            evals += 1
            fit = fitness(res)
            evaluated.append((p, res, fit))
            if better(best_valid, res):
                best_valid = res

        evaluated.sort(key=lambda item: item[2])
        next_population = [item[0] for item in evaluated[:elite_count]]

        if avg_time and evals:
            elapsed = time.time() - t0
            rate = evals / elapsed if elapsed else 0
            remaining = max(est_total - evals, 0)
            eta_left = remaining * avg_time
            print(f"    Stage {stage} progress: {evals:,}/{est_total:,} evals ({rate:.1f} cps), ETA {eta_left:.1f}s")

        while len(next_population) < pop_size:
            p1 = tournament(evaluated, tournament_k, rng)
            p2 = tournament(evaluated, tournament_k, rng)
            child = crossover(p1, p2, stage, rng)
            child = mutate(child, stage, mutation_rate, rng)
            next_population.append(child)
        population = next_population

    return best_valid, evals


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
        return False  # no improvement
    entries[wp] = (volume, row)
    # Rebuild file
    all_rows = [entries[k][1] for k in sorted(entries)]
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(all_rows)
    return True


def main():
    parser = argparse.ArgumentParser(description="Genetic search for gearbox design.")
    parser.add_argument("--wp-start", type=int, default=1200)
    parser.add_argument("--wp-stop", type=int, default=3601)
    parser.add_argument("--wp-step", type=int, default=5)
    parser.add_argument("--pop-size", type=int, default=40)
    parser.add_argument("--generations", type=int, default=60)
    parser.add_argument("--mutation-rate", type=float, default=0.2)
    parser.add_argument("--elite-frac", type=float, default=0.15)
    parser.add_argument("--out-csv", type=str, default="data_genetic.csv")
    parser.add_argument("--skip-3stage", action="store_true")
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
    written = 0
    total_evals = 0
    print(f"Genetic search over {len(pending_wps)} wp values")

    def stage_eval_estimate(stage):
        return args.pop_size * args.generations

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
                best2, evals2 = genetic_stage(
                    wp, 2, args.pop_size, args.generations, args.mutation_rate, args.elite_frac, rng, avg_time=stage_avg.get(2)
                )
                total_evals += evals2
                if best2:
                    print(f"  2-stage best volume: {best2['volume']:.2f}")
                else:
                    print("  2-stage: no valid design")

                best3 = None
                evals3 = 0
                if not args.skip_3stage:
                    best3, evals3 = genetic_stage(
                        wp, 3, args.pop_size, args.generations, args.mutation_rate, args.elite_frac, rng, avg_time=stage_avg.get(3)
                    )
                    total_evals += evals3
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

    print(f"\nTotal evaluations: {total_evals:,}")
    print(f"Wrote {written} designs to {args.out_csv}")


if __name__ == "__main__":
    main()
