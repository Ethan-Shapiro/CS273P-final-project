"""
Creates a small sample dataset for demo purposes.

Takes 20 positive and 30 negative samples from the full training set,
copies their images and metadata into data/sample/ so the demo notebook
can run without the full 401K-image dataset.

Usage:
    python create_sample_data.py
"""

import shutil
from pathlib import Path
import pandas as pd

SEED = 42
N_POS = 20
N_NEG = 30

SRC_CSV = Path("data/train-metadata.csv")
SRC_IMG = Path("data/train-image/image")
DST_DIR = Path("data/sample")
DST_IMG = DST_DIR / "images"
DST_CSV = DST_DIR / "sample_metadata.csv"


def main():
    DST_IMG.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(SRC_CSV)
    print(f"Full dataset: {len(df)} rows  ({df['target'].sum()} positive)")

    df_pos = df[df["target"] == 1].sample(n=min(N_POS, df["target"].sum()), random_state=SEED)
    df_neg = df[df["target"] == 0].sample(n=N_NEG, random_state=SEED)
    df_sample = pd.concat([df_pos, df_neg]).reset_index(drop=True)

    copied = 0
    for isic_id in df_sample["isic_id"]:
        src = SRC_IMG / f"{isic_id}.jpg"
        dst = DST_IMG / f"{isic_id}.jpg"
        if src.exists():
            shutil.copy2(src, dst)
            copied += 1

    df_sample = df_sample[
        df_sample["isic_id"].apply(lambda x: (DST_IMG / f"{x}.jpg").exists())
    ].reset_index(drop=True)

    df_sample.to_csv(DST_CSV, index=False)

    print(f"Sample created in {DST_DIR}/")
    print(f"  Rows: {len(df_sample)}  (pos={df_sample['target'].sum()}, neg={(df_sample['target']==0).sum()})")
    print(f"  Images copied: {copied}")
    print(f"  CSV: {DST_CSV}")


if __name__ == "__main__":
    main()
