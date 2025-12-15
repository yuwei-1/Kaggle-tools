import pandas as pd


def reduce_dataframe_size(df: pd.DataFrame, verbose: bool = False) -> pd.DataFrame:
    bytes_to_kb = 1024
    start_mem = df.memory_usage(deep=True).sum() / bytes_to_kb**2

    large_floats = df.select_dtypes(include=["float64"])
    for col in large_floats:
        df[col] = pd.to_numeric(df[col], downcast="float")

    large_ints = df.select_dtypes(include=["int64"])
    for col in large_ints:
        df[col] = pd.to_numeric(df[col], downcast="integer")

    end_mem = df.memory_usage(deep=True).sum() / bytes_to_kb**2

    if verbose:
        print(f"Initial memory usage: {start_mem:.2f} MB")
        print(f"Reduced memory usage: {end_mem:.2f} MB")
        print(f"Memory reduced by: {(start_mem - end_mem) / start_mem * 100:.1f}%")
    return df
