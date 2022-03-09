""" Run the whole pipeline. """
# pylint: disable=line-too-long
import time
from typing import Any, Dict
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

OUTPUT_DATA: str = DATA_DIR + '/sample_df.csv'
TABLE_NAME: str = 'sample'
OUTPUT_DDL: str = DATA_DIR + '/sample.ddl'
OUTPUT_DDL_TMP: str = DATA_DIR + '/sample.ddl.tmp'

def task_make_ddl() -> Dict[str, Any]:
    """Read the dataframe and make a ddl file, archiving the old one."""
    version: int = 1
    return {
        'actions': [
            (make_ddl.make_ddl, [OUTPUT_DATA, TABLE_NAME, OUTPUT_DDL_TMP]),
            f'[ -f {OUTPUT_DDL_TMP} ]', # make sure it wrote the output file
            f'! [ -f {OUTPUT_DDL} ] || mv {OUTPUT_DDL} {OUTPUT_DDL}{archive_ts()}', # move old one aside
            f'mv {OUTPUT_DDL_TMP} {OUTPUT_DDL}', # move new one into place
            lambda: {VERSION_KEY: version}
        ],
        'file_dep': [OUTPUT_DATA],
        'uptodate': [ (version_unchanged, [version]) ],
        'targets': [OUTPUT_DDL],
        'verbosity': 2
    }

PSQL = 'psql -p 9700 -d "postgres"'

def task_copy_to_db() -> Dict[str, Any]:
    """Archive the old table, create and populate the new one."""
    version: int = 1
    return {
        'actions': [
            f'{PSQL} -c "alter table if exists {TABLE_NAME} rename to {TABLE_NAME}{archive_ts()}"',
            f'{PSQL} -f {OUTPUT_DDL}',
            f'{PSQL} -c "\\copy {TABLE_NAME} from \'{OUTPUT_DATA}\' delimiter \',\' csv header"',
            lambda: {VERSION_KEY: version}
        ],
        'file_dep': [OUTPUT_DDL],
        'uptodate': [ (version_unchanged, [version]) ],
        'verbosity': 2
    }
