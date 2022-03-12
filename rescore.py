"""
Rescore candidates in parallel.

Loads candidates and left and right documents, all in RAM.

Scans the candidates, doing lookups in the left and right,
and sends batches of joined rows to the workers.

TODO: sort and parallel scan the left table and the candidates to avoid
loading the whole left table for lookups.
"""
# pylint: disable=fixme,too-many-arguments,protected-access
import os
import re
import time
from warnings import simplefilter
import multiprocessing
from typing import Callable, List, Optional
import pandas as pd # type:ignore
import psutil # type:ignore
from strsimpy.overlap_coefficient import OverlapCoefficient # type:ignore
from thefuzz import fuzz # type:ignore
import stopwords
import lib

def preprocessor(span: str) -> str:
    """Lower case and remove stopwords, since ngrams don't use stopwords."""
    span = span.lower()
    for stopword in stopwords.STOPWORDS:
        span = re.sub(r'\b' + stopword + r'\b', ' ', span)
    return span

def filtered_read(left_filename: str, column_filter: Optional[List[str]] = None,
                  nrows: Optional[int] = None, skiprows: Optional[int] = None,
                  row_filter_cols: Optional[List[str]] = None,
                  row_filter: Optional[Callable[[pd.DataFrame], bool]] = None) -> pd.DataFrame:
    """ Read the entire table into RAM in the parent, but only the columns and rows specified."""
    left_df = pd.DataFrame()

    usecols: Optional[List[str]] = None
    if column_filter is not None:
        usecols = ['Unnamed: 0'] + column_filter
    if row_filter_cols is not None:
        if usecols is None:
            usecols = row_filter_cols
        else:
            usecols = list(set(usecols).union(set(row_filter_cols)))

    # scan in chunks to avoid filling RAM with rows that will be filtered out
    for left_df_chunk in pd.read_csv(left_filename, engine='c', index_col=0, chunksize = 10000,
                                     dtype='str', nrows=nrows, skiprows=skiprows, low_memory=False,
                                     usecols = usecols):
        left_df_chunk = left_df_chunk.fillna('')
        if row_filter is not None:
            left_df_chunk = left_df_chunk[row_filter]
        left_df_chunk = left_df_chunk[column_filter]
        left_df = pd.concat([left_df, left_df_chunk])
    return left_df

def corrected_similarity(overlap: OverlapCoefficient, left: str, right: str):
    """Overlap Coefficient for two empty strings is 1.0 which will surely confuse the model."""
    if len(left) == 0:
        return 0.0
    if len(right) == 0:
        return 0.0
    return overlap.similarity(left, right)

def worker_fn(chunk: pd.DataFrame,
              scores_columns: List[str],
              scores_filename: str) -> None:
    """Add some columns to the chunk and then write the specified columns.
    TODO: DRY scores_columns somehow, e.g. with a dictionary of callables or something?
    """
    overlap = OverlapCoefficient(k=3)
    for idx, row in chunk.iterrows():
        left_1 = preprocessor(str(row['Supplier']))
        left_2 = preprocessor(str(row['Invoice_Ship_to_Address']))
        right_1 = preprocessor(str(row['Partner_Name']))
        right_2 = preprocessor(str(row['DBA']))

        chunk.at[idx, 'overlap11'] = corrected_similarity(overlap, left_1, right_1)
        chunk.at[idx, 'ratio11'] = fuzz.token_set_ratio(left_1, right_1)/100
        chunk.at[idx, 'overlap12'] = corrected_similarity(overlap, left_1, right_2)
        chunk.at[idx, 'ratio12'] = fuzz.token_set_ratio(left_1, right_2)/100
        chunk.at[idx, 'overlap21'] = corrected_similarity(overlap, left_2, right_1)
        chunk.at[idx, 'ratio21'] = fuzz.token_set_ratio(left_2, right_1)/100
        chunk.at[idx, 'overlap22'] = corrected_similarity(overlap, left_2, right_2)
        chunk.at[idx, 'ratio22'] = fuzz.token_set_ratio(left_2, right_2)/100

    with open(scores_filename, 'a', encoding='utf8') as scores_f:
        chunk.to_csv(scores_f, columns=scores_columns, index=False, header=False)
        scores_f.flush()
    print(f"worker {os.getpid()} finished "
          f"RSS (GB) {psutil.Process(os.getpid()).memory_info().rss/1e9:5.2f} "
          f"start {min(chunk.index):10d} end {max(chunk.index):10d}")

# TODO: externalize these
LEFT_COLS = ['Supplier', 'Invoice_Ship_to_Address']
RIGHT_COLS = ['Partner_Name', 'DBA']
NEW_SCORE_COLS = [
        'overlap11',
        'ratio11',
        'overlap12',
        'ratio12',
        'overlap21',
        'ratio21',
        'overlap22',
        'ratio22'
    ]

def run(candidate_filename: str, chunk_size: int, left_filename: str, right_filename: str,
        scores_filename: str) -> None:
    """Load the decorators in RAM, scan the matches in chunks, decorate the chunks,
    and hand them to the workers for processing."""
    simplefilter(action="ignore", category=pd.errors.PerformanceWarning)
    left_table: pd.DataFrame = filtered_read(left_filename, column_filter=LEFT_COLS, nrows=100000,
                                             skiprows=None, row_filter_cols=['Fiscal_Year'],
                                             row_filter=lambda x: x['Fiscal_Year']=='2020')
    right_table: pd.DataFrame = filtered_read(right_filename, column_filter=RIGHT_COLS)

    candidate_columns: List[str] = list(pd.read_csv(candidate_filename, nrows=0).columns)
    scores_columns: List[str] = candidate_columns + NEW_SCORE_COLS
    lib.write_header(scores_filename, scores_columns)

    ctx = multiprocessing.get_context('spawn')
    with ctx.Pool(processes = 6) as pool:
        # candidates is the driving table
        rows_read = 0
        for chunk in pd.read_csv(candidate_filename, chunksize=chunk_size, nrows=2000000):
            rows_read += chunk_size
            print(f"parent rows {rows_read:10d}")
            # decorate the candidates with left and right data
            chunk = chunk.merge(left_table, left_on='left_index', right_index=True, how='left')
            chunk = chunk.merge(right_table, left_on='right_index', right_index=True, how='left')
            chunk = chunk[LEFT_COLS + RIGHT_COLS + candidate_columns]

            pool.apply_async(worker_fn, (chunk, scores_columns, scores_filename))
            # avoid over-filling the queue
            while pool._taskqueue.qsize() > 12: # type:ignore
                time.sleep(1)
        pool.close()
        pool.join()

if __name__ == '__main__':
    run('sample-data/sample-candidates.csv',
        10000,
        'sample-data/sample-left.csv',
        'sample-data/sample-right.csv',
        'sample-data/sample-scores.csv')
