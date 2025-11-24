import itertools
import pandas as pd
import numpy

# 1. Define your variables and ranges
# Each variable can be:
#   - a list of values, OR
#   - a range generated with range(start, stop, step)
variables = {
    "n1": numpy.arange(1, 9.1, 0.1),             # Stage 1 Input Ratio
    "Pdn1": [4, 5, 6, 8, 10],                    # Diametral Pitch
    "Np1": range(10, 101, 1),                    # Pinion 1 Teeth Number
    "Helix1": range(15, 26, 5),                  # Helix angle, for now we're assuming both stages have the same helix angle.
    "Pdn2": [4, 5, 6, 8, 10],                    # Diametral Pitch Stage 2
    "Np2": range(10, 101, 1),                    # Pinion 2 Teeth Number
#    "Helix2": range(15, 26, 5),                  # Helix angle Stage 2
}

# 2. Generate all combinations
keys = list(variables.keys())
values = [list(v) for v in variables.values()]  # ensure all are lists
combinations = list(itertools.product(*values))

# 3. Convert to DataFrame
df = pd.DataFrame(combinations, columns=keys)

# 4. Save to CSV
output_file = "all_combinations.csv"
df.to_csv(output_file, index=False)

print(f"Generated {len(df)} combinations → saved to {output_file}")


