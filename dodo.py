""" Run the whole pipeline. """
# pylint: disable=line-too-long
import time
from typing import Any, Dict
import classify
import make_ddl

DOIT_CONFIG: Dict[str, str] = {
    'backend': 'json',
    'check_file_uptodate': 'timestamp'
}

VERSION_KEY: str = 'version'
DATA_DIR: str = 'sample-data'

def version_unchanged(task, values, version) -> bool: # pylint: disable=unused-argument
    """True if the previous version is the same as the new version"""
    return values.get(VERSION_KEY) == version

def archive_ts() -> str:
    """A string for archived versions"""
    return time.strftime("%Y%m%d%H%M%S")

SCORE_FILE = DATA_DIR + '/sample-scores.csv'
CHUNK_SIZE = 10000
MODEL_FILE = DATA_DIR + '/sample-model.pkl'
THRESHOLD = 0.5
PREDICTION_FILE = DATA_DIR + '/sample-predictions.csv'

def task_classify() -> Dict[str, Any]:
    """Read scores, classify with model, write predictions."""
    version: int = 4
    return {
        'actions': [
            (classify.run, [SCORE_FILE, CHUNK_SIZE, MODEL_FILE, THRESHOLD, PREDICTION_FILE]),
            lambda: {VERSION_KEY: version}
        ],
        'file_dep': [SCORE_FILE, MODEL_FILE],
        'targets': [PREDICTION_FILE],
        'uptodate': [ (version_unchanged, [version]) ],
        'verbosity': 2
    }

PREDICTION_TABLE: str = 'sample_predictions'
PREDICTION_DDL: str = DATA_DIR + '/sample-predictions.ddl'
PREDICTION_DDL_TMP: str = DATA_DIR + '/sample-predictions.ddl.tmp'

def task_make_ddl() -> Dict[str, Any]:
    """Read the dataframe and make a ddl file, archiving the old one."""
    version: int = 1
    return {
        'actions': [
            (make_ddl.run, [PREDICTION_FILE, PREDICTION_TABLE, PREDICTION_DDL_TMP]),
            f'[ -f {PREDICTION_DDL_TMP} ]', # make sure it wrote the output file
            f'! [ -f {PREDICTION_DDL} ] || mv {PREDICTION_DDL} {PREDICTION_DDL}{archive_ts()}', # move old one aside
            f'mv {PREDICTION_DDL_TMP} {PREDICTION_DDL}', # move new one into place
            lambda: {VERSION_KEY: version}
        ],
        'file_dep': [PREDICTION_FILE],
        'targets': [PREDICTION_DDL],
        'uptodate': [ (version_unchanged, [version]) ],
        'verbosity': 2
    }

PSQL = 'psql -p 9700 -d "postgres"'

def task_copy_to_db() -> Dict[str, Any]:
    """Archive the old table, create and populate the new one."""
    version: int = 1
    return {
        'actions': [
            f'{PSQL} -c "alter table if exists {PREDICTION_TABLE} rename to {PREDICTION_TABLE}{archive_ts()}"',
            f'{PSQL} -f {PREDICTION_DDL}',
            f'{PSQL} -c "\\copy {PREDICTION_TABLE} from \'{PREDICTION_FILE}\' delimiter \',\' csv header"',
            lambda: {VERSION_KEY: version}
        ],
        'file_dep': [PREDICTION_DDL],
        'uptodate': [ (version_unchanged, [version]) ],
        'verbosity': 2
    }
