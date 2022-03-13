""" Run the whole pipeline. """
# pylint: disable=line-too-long
import time
from typing import Any, Dict
import candidates, fit, rescore, classify, make_ddl

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

LEFT_FILE = DATA_DIR + '/sample-left.csv'
RIGHT_FILE = DATA_DIR + '/sample-right.csv'
CANDIDATES_FILE = DATA_DIR + '/sample-candidates.csv'

def task_candidates() -> Dict[str, Any]:
    """Read right and left, generate candidate pairs."""
    version: int = 1
    return {
        'actions': [
            (candidates.run, [LEFT_FILE, RIGHT_FILE, CANDIDATES_FILE]),
            lambda: {VERSION_KEY: version}
        ],
        'file_dep': [LEFT_FILE, RIGHT_FILE],
        'targets': [CANDIDATES_FILE],
        'uptodate': [ (version_unchanged, [version]) ],
        'verbosity': 2
    }

LABEL_FILE = DATA_DIR + '/sample-labeled-scores.csv'
MODEL_FILE = DATA_DIR + '/sample-model.pkl'

def task_fit() -> Dict[str, Any]:
    """Read labeled training, fit a model, and save it."""
    version: int = 1
    return {
        'actions': [
            (fit.run, [LABEL_FILE, MODEL_FILE]),
            lambda: {VERSION_KEY: version}
        ],
        'file_dep': [LABEL_FILE],
        'targets': [MODEL_FILE],
        'uptodate': [ (version_unchanged, [version]) ],
        'verbosity': 2
    }

CHUNK_SIZE = 10000
SCORE_FILE = DATA_DIR + '/sample-scores.csv'

def task_rescore() -> Dict[str, Any]:
    """Read candidates, decorate, score again, write all scores."""
    version: int = 1
    return {
        'actions': [
            (rescore.run, [CANDIDATES_FILE, CHUNK_SIZE, LEFT_FILE, RIGHT_FILE, SCORE_FILE]),
            lambda: {VERSION_KEY: version}
        ],
        'file_dep': [CANDIDATES_FILE, LEFT_FILE, RIGHT_FILE],
        'targets': [SCORE_FILE],
        'uptodate': [ (version_unchanged, [version]) ],
        'verbosity': 2
    }


SCORE_FILE = DATA_DIR + '/sample-scores.csv'
THRESHOLD = 0.5
PREDICTION_FILE = DATA_DIR + '/sample-predictions.csv'

def task_classify() -> Dict[str, Any]:
    """Read scores, classify with model, write predictions."""
    version: int = 1
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
