"""
Generate record-linking candidates using various fields.

Broadcast the (smaller) right table, scan the (big) left table in chunks and send
the chunks to workers.

On my 6-core i5-9400F machine, this scans at ~500 rows/sec, against the broadcast table of 6K rows,
so the whole 22M row left table scan would take about 12 hours.
"""
# pylint: disable=line-too-long,fixme,protected-access,invalid-name,global-statement
import re
import os
import time
import multiprocessing
from typing import List
from warnings import simplefilter
import psutil # type:ignore
import pandas as pd # type:ignore
from red_string_grouper import record_linkage # type:ignore
import usaddress # type:ignore
import stopwords
import lib

def extract_recipient(addr_str) -> str:
    """ Extract useful terms embedded in addresses."""
    parsed_addr = usaddress.parse(addr_str)
    output: List[str] = []
    for token, label in parsed_addr:
        if len(output) > 0 and label != 'Recipient':
            break
        if label == 'Recipient':
            output.append(token)
    return " ".join(output)

def preprocessor(text_field: str) -> str:
    """ Simple field transforms: lower, stopwords, addr parsing."""
    text_field = text_field.lower()
    text_field = extract_recipient(text_field) # Removes stuff like "po box 1234 santa monica ca."
    for stopword in stopwords.STOPWORDS:
        text_field = re.sub(r'\b' + stopword + r'\b', ' ', text_field)
    return text_field

# global
right_df = None
def worker_init(right_filename) -> None:
    """ Each worker loads the right file once, i.e. "broadcast join." """
    global right_df
    right_df = pd.read_csv(right_filename, engine='c', index_col=0,
                                  usecols=['Unnamed: 0', 'Partner_Name', 'DBA'],
                                  low_memory=False).fillna('')

def worker_fn(left_df: pd.DataFrame, output_filename: str) -> int:
    """ Process the given chunk of the left table.  Returns the number of rows written. """
    simplefilter(action="ignore", category=UserWarning)
    pd.set_option('mode.chained_assignment', None)

    left_df = left_df.fillna('')

    matches = record_linkage(
        data_frames = [left_df, right_df],
        fields_2b_matched_fuzzily = [
            ('Supplier','Partner_Name',1.0, {'min_similarity':0.2,'ngram_size':[1,2],'analyzer':'word', 'preprocessor':preprocessor}),
            ('Supplier','Partner_Name',0.2, {'min_similarity':0.2,'ngram_size':[5,7],'analyzer':'char_wb', 'preprocessor':preprocessor}),
            ('Supplier','DBA',0.8, {'min_similarity':0.2,'ngram_size':[1,2],'analyzer':'word', 'preprocessor':preprocessor}),
            ('Supplier','DBA',0.7, {'min_similarity':0.2,'ngram_size':[5,7],'analyzer':'char_wb', 'preprocessor':preprocessor}),
            ('Invoice_Ship_to_Address','Partner_Name',0.3, {'min_similarity':0.2,'ngram_size':[1,2],'analyzer':'word', 'preprocessor':preprocessor}),
            ('Invoice_Ship_to_Address','Partner_Name',0.25, {'min_similarity':0.2,'ngram_size':[5,7],'analyzer':'char_wb', 'preprocessor':preprocessor}),
            ('Invoice_Ship_to_Address','DBA',0.5, {'min_similarity':0.2,'ngram_size':[1,2],'analyzer':'word', 'preprocessor':preprocessor}),
            ('Invoice_Ship_to_Address','DBA',0.1, {'min_similarity':0.2,'ngram_size':[5,7],'analyzer':'char_wb', 'preprocessor':preprocessor}),
        ],
        fields_2b_matched_exactly = [],
        hierarchical=False, # Non-hierarchical output just keeps the scores.
        stop_words=stopwords.STOPWORDS,
        binary=False,  # Allow term weighting (by repetition). TODO: actually do that?
        n_blocks=(1,1) # Don't let the string grouper divide anything up, it doesn't seem to help anyway.
    )

    filtered_df = matches[matches['Weighted Mean Similarity Score']>0.02]

    #filtered_df.to_csv(candidates_file)

    with open(output_filename, 'a', encoding='utf8') as output_f:
        filtered_df.to_csv(output_f, header=False)
        output_f.flush()

    print(f"worker {os.getpid()} finished "
          f"RSS (GB) {psutil.Process(os.getpid()).memory_info().rss/1e9:5.2f} ")

    return len(filtered_df)

# TODO: do this differently
CANDIDATE_COLS = ['left_index', 'right_index', 'Weighted Mean Similarity Score',
       '0:Supplier/Partner_Name', '1:Supplier/Partner_Name', '2:Supplier/DBA',
       '3:Supplier/DBA', '4:Invoice_Ship_to_Address/Partner_Name',
       '5:Invoice_Ship_to_Address/Partner_Name',
       '6:Invoice_Ship_to_Address/DBA', '7:Invoice_Ship_to_Address/DBA']

def run(left_file: str, right_file: str, candidates_file: str) -> None:
    """Do everything."""
    simplefilter(action="ignore", category=pd.errors.PerformanceWarning)
    pd.set_option('mode.chained_assignment', None)

    lib.write_header(candidates_file, CANDIDATE_COLS)

    chunksize  = 100000
    ctx = multiprocessing.get_context('spawn')
    # TODO: read the right file in the pool worker initializer
    with ctx.Pool(processes = 6, initializer = worker_init, initargs = (right_file, )) as pool:
        # TODO row filter, e.g. df = df[df['Fiscal_Year']=='2020']
        # TODO does this need encoding='latin1' or na_filter=False?
        # TODO skiprows?
        rows_read = 0
        for left_df_chunk in pd.read_csv(left_file, engine='c', index_col=0, dtype='str', chunksize=chunksize, nrows=1000000,
                                          low_memory=False, usecols=['Unnamed: 0', 'Supplier','Invoice_Ship_to_Address']):
            rows_read += len(left_df_chunk)
            print(f"parent rows read: {rows_read:10d}")
            pool.apply_async(worker_fn, (left_df_chunk, candidates_file),
                             callback=lambda x: print(f"worker rows written: {x:10d}"),
                             error_callback=lambda x: print(f"error: {x}"))
            while pool._taskqueue.qsize() > 3: # type:ignore
                time.sleep(1)
        pool.close()
        pool.join()

if __name__ == '__main__':
    run('sample-data/sample-left.csv', 'sample-data/sample-right.csv', 'sample-data/sample-candidates.csv')
