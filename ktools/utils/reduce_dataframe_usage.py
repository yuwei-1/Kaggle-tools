import pandas as pd


def reduce_dataframe_size(df):
    
    bytes_to_kb = 1024
    start_mem = df.memory_usage(deep=True).sum() / bytes_to_kb ** 2
    print(f"Initial memory usage: {start_mem:.2f} MB")

    for col in df.columns:
        col_type = df[col].dtypes
        
        if col_type in {'float64', 'float32'}:
            df[col] = pd.to_numeric(df[col], downcast='float')
        elif col_type in {'int64', 'int32'}:
            df[col] = pd.to_numeric(df[col], downcast='integer')
        else:
            continue

    end_mem = df.memory_usage(deep=True).sum() / bytes_to_kb ** 2
    print(f"Reduced memory usage: {end_mem:.2f} MB")
    print(f"Memory reduced by: {(start_mem - end_mem) / start_mem * 100:.1f}%")
    return df