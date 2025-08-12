import pandas as pd
import logging

from pathlib import Path
from typing import List

# Configuration constants
VERIFICATION_DIR = Path("../1_verification")
OUTPUT_DIR       = Path("datasources")
OUTPUT_FILE      = "vr.parquet"


def load_datasource(
    db: str,
    base_dir: Path = VERIFICATION_DIR,
    ext: str = ".parquet",
    engine: str = "pyarrow"
) -> pd.DataFrame:
    """
    Load a parquet file located at <base_dir>/<db>/vrfcs.parquet,
    add a column 'db' indicating the source, and return the DataFrame.
    """
    path = base_dir / db / f"vrfcs{ext}"
    if not path.exists():
        logging.error(f"File not found: {path}")
        raise FileNotFoundError(f"No such file: {path}")

    df = pd.read_parquet(path, engine=engine)
    df['db'] = db
    return df


def merge_datasources(dbs: List[str]) -> pd.DataFrame:
    """
    Load and concatenate multiple datasources, each tagged with its 'db' identifier.
    """
    dfs = [load_datasource(db) for db in dbs]
    return pd.concat(dfs, ignore_index=True)


if __name__ == "__main__":
    dbs = ["gcm", "ft"]
    merged_df = merge_datasources(dbs)

    # Ensure output directory exists
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Write out with snappy compression
    merged_df.to_parquet(
        OUTPUT_DIR / OUTPUT_FILE,
        engine="pyarrow",
        compression="snappy"
    )

    print(merged_df.head())
    print(f"Total rows × columns: {merged_df.shape}")