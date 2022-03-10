"""
Classify the candidates with a parallel map.

On my 6-core machine, runs about 1.2M row/sec
"""
from multiprocessing import Pool
from multiprocessing import set_start_method
import pickle
from pygam import LogisticGAM # type:ignore
import pandas as pd # type:ignore

def write_header(output_file: str) -> None:
    """Write the csv header so that the workers can append to it."""
    with open(output_file, 'w', encoding='utf8') as output_header_f :
        result_df = pd.DataFrame(columns=['left_index','right_index', 'prediction_prob'])
        result_df.to_csv(output_header_f, index=False)
        output_header_f.flush()

def worker_fn(model_file: str, threshold: float,
              scores_chunk_df: pd.DataFrame, output_file: str) -> None:
    """Accept a chunk from the main reader, map each row, and append to the output."""
    x_data = scores_chunk_df.drop(columns=scores_chunk_df.columns[0:2])
    predictions = scores_chunk_df.loc[:,['left_index','right_index']]
    del scores_chunk_df
    with open(model_file, 'rb') as model_f:
        model: LogisticGAM = pickle.load(model_f)
    predictions['prediction_prob'] = model.predict_proba(x_data.values)
    predictions = predictions[predictions['prediction_prob'] > threshold]
    with open(output_file, 'a', encoding='utf8') as output_f:
        predictions.to_csv(output_f, float_format='%.4f', index=False, header=False)
        output_f.flush()

def run(input_file: str, chunk_size: int,
        model_file: str, threshold: float,
        output_file: str) -> None:
    """Spawn some workers and do classifications."""
    set_start_method('spawn')
    write_header(output_file)

    with Pool(processes = 6) as pool:
        # use apply() instead of map() because map exhausts the iterator to find length
        for chunk in pd.read_csv(input_file, index_col=0, chunksize=chunk_size):
            pool.apply_async(worker_fn, (model_file, threshold, chunk, output_file))
        pool.close()
        pool.join()

if __name__ == '__main__':
    run('sample-data/scores.csv', 10000,
        'sample-data/sample-model.pkl', 0.5,
        'sample-data/predictions.csv')
