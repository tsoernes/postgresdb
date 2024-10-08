from typing import Any, Collection, Optional, Union

import numpy as np
import pandas as pd


def blank_str_to_nan(
    df, cols: Optional[Union[str, Collection[str]]] = None, nan=np.nan
) -> None:
    """
    Set empty or just whitespace string entries in 'col' to NaN, inplace
    """
    if isinstance(df, pd.Series):
        df.replace(r"^\s*$", nan, regex=True, inplace=True)
        return
    if cols is None:
        cols = (
            df.convert_dtypes(convert_integer=False, convert_boolean=False)
            .select_dtypes("string")
            .columns
        )
    elif isinstance(cols, str):
        cols = (cols,)
    for col in cols:
        if str(df[col].dtype) == "string":
            # NOTE Workaround for bug in Pandas that fails to replace if dtype is String
            df[col] = (
                df[col]
                .astype("object")
                .replace(r"^\s*$", pd.NA, regex=True)
                .astype("string")
            )
        else:
            df[col].replace(r"^\s*$", nan, regex=True, inplace=True)


def nan_to_none(val) -> Any:
    """
    >>> nan_to_none(np.nan) is None
    True
    >>> nan_to_none(pd.NA) is None
    True
    >>> nan_to_none(pd.NaT) is None
    True
    >>> nan_to_none('') is None
    False
    >>> nan_to_none([]) == []
    True
    """
    if (
        not isinstance(val, Collection)
        and not isinstance(val, pd.Series)
        and not isinstance(val, pd.DataFrame)
        and pd.isna(val)
    ):
        return None
    return val
