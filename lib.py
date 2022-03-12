"""A few common functions"""

from typing import List
import pandas as pd # type:ignore

def write_header(output_file: str, columns: List[str]) -> None:
    """Write the csv header so that the workers can append to it."""
    with open(output_file, 'w', encoding='utf8') as output_header_f:
        result_df = pd.DataFrame(columns=columns)
        result_df.to_csv(output_header_f, index=False)
        output_header_f.flush()
