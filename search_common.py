import itertools
import random
import time
import numpy
import functools
import calculations as c

# Shared limits
MAX_BENDING_STRESS = 36.8403  # ksi
MAX_CONTACT_STRESS = 129.242  # ksi

# Search space (mirrors data_generator.py)
VARIABLES = {
    "wp": range(1200, 3601, 10),
    "n1": numpy.arange(1, 9.1, 0.1),
    "n2": numpy.arange(1, 9.1, 0.1),
    "Pd1": [4, 5, 6, 8, 10],
    "Pd2": [4, 5, 6, 8, 10],
    "Pd3": [4, 5, 6, 8, 10],
    "Np1": range(10, 101, 2),
    "Np2": range(10, 101, 2),
    "Np3": range(10, 101, 2),
    "Helix1": [15, 20, 25],
    "Helix2": [15, 20, 25],
    "Helix3": [15, 20, 25],
}

STAGE_KEYS = {
    2: ["n1", "Pd1", "Np1", "Helix1", "Pd2", "Np2", "Helix2"],
    3: ["n1", "Pd1", "Np1", "Helix1", "n2", "Pd2", "Np2", "Helix2", "Pd3", "Np3", "Helix3"],
}

CSV_HEADER = [
    "wp",
    "wf",
    "P",
    "stages",
    "volume",
    "n1",
    "Pd1",
    "Np1",
    "Helix1",
    "n2",
    "Pd2",
    "Np2",
    "Helix2",
    "Pd3",
    "Np3",
    "Helix3",
    "s1_bending",
    "s1_contact",
    "s2_bending",
    "s2_contact",
    "s3_bending",
    "s3_contact",
]


def choice_list(key):
    """Return a list of available choices for the given key."""
    values = VARIABLES[key]
    try:
        return list(values)
    except Exception:
        return [v for v in values]


def snap_to_choice(value, key):
    """Snap a value to the closest available choice for this key."""
    choices = choice_list(key)
    return min(choices, key=lambda x: abs(x - value))


def neighborhood(key, center, radius):
    """Return a neighborhood of choices around center (inclusive)."""
    choices = choice_list(key)
    if not choices:
        return []
    closest = snap_to_choice(center, key)
    idx = choices.index(closest)
    start = max(0, idx - radius)
    end = min(len(choices), idx + radius + 1)
    return choices[start:end]


def random_params(stage, rng=None):
    """Sample random parameter set for a stage."""
    rng = rng or random
    params = {}
    for key in STAGE_KEYS[stage]:
        params[key] = rng.choice(choice_list(key))
    return params


def is_valid(stresses, stage):
    """Check stress limits for a given stage count."""
    base_ok = (
        stresses["s1_bending"] is not None
        and stresses["s1_contact"] is not None
        and stresses["s2_bending"] is not None
        and stresses["s2_contact"] is not None
        and stresses["s1_bending"] <= MAX_BENDING_STRESS
        and stresses["s1_contact"] <= MAX_CONTACT_STRESS
        and stresses["s2_bending"] <= MAX_BENDING_STRESS
        and stresses["s2_contact"] <= MAX_CONTACT_STRESS
    )
    if stage == 2:
        return base_ok
    return (
        base_ok
        and stresses.get("s3_bending") is not None
        and stresses.get("s3_contact") is not None
        and stresses["s3_bending"] <= MAX_BENDING_STRESS
        and stresses["s3_contact"] <= MAX_CONTACT_STRESS
    )


def evaluate_design(wp, params, stage):
    """Evaluate a design; return details or None on failure."""
    try:
        if stage == 2:
            wf, P, volume, s1b, s1c, s2b, s2c, _, _ = c.results(
                wp,
                params["n1"],
                None,
                params["Pd1"],
                params["Np1"],
                params["Helix1"],
                params["Pd2"],
                params["Np2"],
                params["Helix2"],
                None,
                None,
                None,
            )
            stresses = {
                "s1_bending": s1b,
                "s1_contact": s1c,
                "s2_bending": s2b,
                "s2_contact": s2c,
                "s3_bending": None,
                "s3_contact": None,
            }
        else:
            wf, P, volume, s1b, s1c, s2b, s2c, s3b, s3c = c.results(
                wp,
                params["n1"],
                params["n2"],
                params["Pd1"],
                params["Np1"],
                params["Helix1"],
                params["Pd2"],
                params["Np2"],
                params["Helix2"],
                params["Pd3"],
                params["Np3"],
                params["Helix3"],
            )
            stresses = {
                "s1_bending": s1b,
                "s1_contact": s1c,
                "s2_bending": s2b,
                "s2_contact": s2c,
                "s3_bending": s3b,
                "s3_contact": s3c,
            }
        valid = is_valid(stresses, stage)
        return {
            "wp": wp,
            "wf": wf,
            "P": P,
            "stages": stage,
            "volume": volume,
            "params": params,
            "stresses": stresses,
            "valid": valid,
        }
    except Exception:
        return None


def better(a, b):
    """Return True if b is a better design than a."""
    if b is None or not b.get("valid", False):
        return False
    if a is None:
        return True
    return b["volume"] < a["volume"]


def design_row(design):
    """Convert evaluated design to CSV row."""
    p = design["params"]
    s = design["stresses"]
    return [
        design["wp"],
        design["wf"],
        design["P"],
        design["stages"],
        design["volume"],
        p.get("n1"),
        p.get("Pd1"),
        p.get("Np1"),
        p.get("Helix1"),
        p.get("n2"),
        p.get("Pd2"),
        p.get("Np2"),
        p.get("Helix2"),
        p.get("Pd3"),
        p.get("Np3"),
        p.get("Helix3"),
        s.get("s1_bending"),
        s.get("s1_contact"),
        s.get("s2_bending"),
        s.get("s2_contact"),
        s.get("s3_bending"),
        s.get("s3_contact"),
    ]


def random_valid_design(wp, stage, attempts=200, rng=None):
    """Try random samples until a valid design is found or attempts run out."""
    rng = rng or random
    best = None
    for _ in range(attempts):
        params = random_params(stage, rng)
        res = evaluate_design(wp, params, stage)
        if res and res["valid"]:
            return res
        if better(best, res):
            best = res
    return best


def sample_eval_time(stage, samples=50, rng=None):
    """Measure average seconds per evaluate_design call for a given stage."""
    rng = rng or random
    total = 0.0
    attempted = 0
    for _ in range(samples):
        params = random_params(stage, rng)
        wp = rng.choice(choice_list("wp"))
        start = time.time()
        try:
            _ = evaluate_design(wp, params, stage)
        except Exception:
            pass
        total += (time.time() - start)
        attempted += 1
    if attempted == 0:
        return None
    return total / attempted


def make_cached_evaluator(maxsize=200000):
    """
    Return a cached evaluate function with LRU eviction.
    Cache key is (stage, wp, ordered params tuple).
    """
    keys_by_stage = {stage: tuple(STAGE_KEYS[stage]) for stage in STAGE_KEYS}

    @functools.lru_cache(maxsize=maxsize)
    def _cached(stage, wp, param_tuple):
        keys = keys_by_stage[stage]
        params = dict(zip(keys, param_tuple))
        return evaluate_design(wp, params, stage)

    def wrapper(wp, params, stage):
        key_tuple = tuple(params[k] for k in keys_by_stage[stage])
        return _cached(stage, wp, key_tuple)

    wrapper.cache_clear = _cached.cache_clear  # type: ignore[attr-defined]
    return wrapper
