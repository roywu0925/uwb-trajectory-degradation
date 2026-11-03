"""Step 0: look at what you actually downloaded.

Run this FIRST, before anything else:

    python inspect_data.py

It prints every csv under data/, its shape, its columns and a couple of
rows. Paste the output back into the chat and we lock the column map in
config.py -- after that nothing else in the pipeline has to guess.
"""

import sys

import pandas as pd

from config import DATA_DIR

PREVIEW_ROWS = 3
MAX_BYTES_FOR_FULL_READ = 200_000_000


def describe(path):
    size_mb = path.stat().st_size / 1e6
    print("=" * 72)
    print(f"{path}   ({size_mb:.1f} MB)")
    print("=" * 72)

    nrows = None if path.stat().st_size < MAX_BYTES_FOR_FULL_READ else 200_000
    df = pd.read_csv(path, nrows=nrows)
    if nrows:
        print(f"[read first {nrows:,} rows only -- file is large]")

    print(f"shape: {df.shape[0]:,} rows x {df.shape[1]} cols\n")
    print("columns:")
    for col in df.columns:
        n_unique = df[col].nunique(dropna=True)
        sample = df[col].dropna().iloc[0] if df[col].notna().any() else "-"
        print(f"  {col:<24} {str(df[col].dtype):<10} "
              f"unique={n_unique:<8} e.g. {sample}")

    print(f"\nfirst {PREVIEW_ROWS} rows:")
    with pd.option_context("display.max_columns", None, "display.width", 200):
        print(df.head(PREVIEW_ROWS))
    print()


def main():
    if not DATA_DIR.exists():
        sys.exit(f"{DATA_DIR} does not exist -- unzip the Kaggle download there.")

    paths = sorted(DATA_DIR.rglob("*.csv"))
    if not paths:
        sys.exit(f"no csv files found under {DATA_DIR}")

    print(f"found {len(paths)} csv file(s) under {DATA_DIR}\n")
    for path in paths:
        try:
            describe(path)
        except Exception as exc:
            print(f"!! could not read {path}: {exc}\n")


if __name__ == "__main__":
    main()
