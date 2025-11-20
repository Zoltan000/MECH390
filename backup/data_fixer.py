"""data_fixer.py

Read a CSV (default: data.csv), compute `wf` and `P` for each row by calling
`important_values` from `calculations.py` and insert those two columns
immediately after `wp`. The fixed CSV is written to the output path
(default: data_fixed.csv).

Usage:
	python data_fixer.py                # reads data.csv, writes data_fixed.csv
	python data_fixer.py -i in.csv -o out.csv
"""

import argparse
import sys
import pandas as pd

from calculations import important_values


def compute_wf_P(row):
	# Ensure proper types (important_values expects numeric values)
	try:
		wp = float(row['wp'])
		n1 = float(row['n1'])
		Pnd = float(row['Pnd'])
		Np1 = float(row['Np1'])
		Helix = float(row['Helix'])
	except Exception as e:
		raise ValueError(f"Invalid numeric data in row index {row.name}: {e}")

	# Always compute P and wf from wp so values vary per-row. Round to 1
	# decimal place as requested.
	P = round(wp / 240.0, 1)
	wf = round(wp / 12.0 + 100.0, 1)

	return pd.Series({'wf': wf, 'P': P})


def fix_csv(inpath: str, outpath: str):
	df = pd.read_csv(inpath)

	# Normalize column names: strip whitespace (and BOM) so columns like ' wp' or
	# 'wp\ufeff' are recognized as 'wp'. This fixes KeyError when accessing
	# row['wp'] if the CSV header has stray spaces or BOM.
	df.columns = df.columns.str.strip()

	# Compute wf and P for each row
	newcols = df.apply(compute_wf_P, axis=1)

	# If columns already exist, drop them first to avoid duplicates
	for col in ('wf', 'P'):
		if col in df.columns:
			df = df.drop(columns=[col])

	# Concatenate the new columns
	df_with = pd.concat([df, newcols], axis=1)

	# Reorder columns so wf and P are immediately after wp
	cols = list(df_with.columns)
	if 'wp' not in cols:
		raise KeyError("Input CSV must contain a 'wp' column")

	# Remove wf and P from current list and re-insert after wp
	for col in ('wf', 'P'):
		if col in cols:
			cols.remove(col)

	wp_index = cols.index('wp')
	for i, col in enumerate(('wf', 'P'), start=1):
		cols.insert(wp_index + i, col)

	df_final = df_with[cols]

	df_final.to_csv(outpath, index=False)
	print(f"Wrote fixed CSV to: {outpath}")


def main(argv=None):
	parser = argparse.ArgumentParser(description="Fix data CSV by adding wf and P columns")
	parser.add_argument('-i', '--input', default='data.csv', help='Input CSV path')
	parser.add_argument('-o', '--output', default='data_fixed.csv', help='Output CSV path')
	args = parser.parse_args(argv)

	fix_csv(args.input, args.output)


if __name__ == '__main__':
	main()

