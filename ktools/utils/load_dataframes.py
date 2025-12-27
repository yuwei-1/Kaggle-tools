from pathlib import Path
import pandas as pd
import polars as pl


def load_all_dataframes(dir_path: str) -> pd.DataFrame:
    """Load all CSV files from a directory and concatenate them column-wise (horizontally).

    Args:
        dir_path: Path to directory containing CSV files.

    Returns:
        A LazyFrame with all CSV files concatenated horizontally.
    """
    dir_path: Path = Path(dir_path)

    csv_files = sorted(dir_path.glob("*.csv"))
    lazy_frames = [pl.scan_csv(f).select(pl.all().exclude("")) for f in csv_files]
    lazy_frame = pl.concat(lazy_frames, how="horizontal")
    return lazy_frame.collect().to_pandas()
