import pm4py
import pandas as pd

def read_log(path: str, log: str) -> pd.DataFrame:
    """
    Reads event log and preprocess it

    Parameters
    -----------------------
    path: str,
        File path
    log: str,
        File name
    Returns
    -----------------------
    Processed event log containing the only the necessary columns for encoding
    """

    return pm4py.read_xes(f"{path}/{log}")
