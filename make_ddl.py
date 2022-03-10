"""Read a dataframe from csv and make a corresponding create table ddl file."""
# pylint: disable=line-too-long
from typing import List
import numpy as np
import pandas as pd # type:ignore

def dtypes_reduce(data_frame, coltypes: List[str]) -> List[str]:
    """Map dataframe types to postgres types"""
    if len(coltypes) == 0:
        coltypes = [''] * len(data_frame.dtypes)
    for idx, dtype in enumerate(data_frame.dtypes):
        if coltypes[idx] == 'TEXT':
            continue
        if dtype == object:
            coltypes[idx] = 'TEXT'
        elif dtype in (np.float64, np.int64):
            coltypes[idx] = 'NUMERIC'
        else:
            raise ValueError(f"weird type: {dtype}")
    return coltypes

def run(input_filename: str, tablename: str, output_filename: str):
    """Read the csv, write the ddl."""
    coltypes: List[str] = []
    df_sample = None
    for df_chunk in pd.read_csv(input_filename, chunksize=10):
        if df_sample is None:
            df_sample = df_chunk
        coltypes = dtypes_reduce(df_chunk, coltypes)
    if df_sample is None:
        raise Exception(f"No data found in {input_filename}")
    with open(output_filename, 'w', encoding='utf8') as output_file:
        output_file.write(pd.io.sql.get_schema(df_sample,
                                               tablename,
                                               dtype=dict(zip(df_sample.columns, coltypes))))
        output_file.write("\n")

if __name__ == '__main__':
    run('sample-data/sample_df.csv', 'sample', 'sample-data/sample.ddl')
