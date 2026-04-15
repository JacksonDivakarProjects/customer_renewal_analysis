import re
import pandas as pd

def clean_yes_no_column(df, column_name, new_column_name=None, inplace=False):
    """
    Normalize a column containing yes/no responses into a standardized format.

    The function strips extraneous characters (like markdown bold and brackets), converts
    to lowercase, and maps variations of "yes"/"no" to "Yes"/"No". Values that are ambiguous,
    missing, or do not start with "yes"/"no" are replaced with "N/A". The special value
    "not applicable" is left unchanged.

    Args:
        df (pd.DataFrame): Input DataFrame.
        column_name (str): Name of the column to clean.
        new_column_name (str, optional): Name for the cleaned column. If None, the original
            column is overwritten. Defaults to None.
        inplace (bool, optional): If True, modify the original DataFrame; otherwise, return
            a copy. Defaults to False.

    Returns:
        pd.DataFrame: DataFrame with the cleaned column (either modified in place or a copy).

    Examples:
        >>> import pandas as pd
        >>> df = pd.DataFrame({"response": ["**yes**", "[no]", "Yes, please", "N/A", ""]})
        >>> clean_yes_no_column(df, "response")
          response
        0      Yes
        1       No
        2      N/A
        3      N/A
        4      N/A
    """
    
    if not inplace:
        df = df.copy()

    def _clean_single(value):
        # Convert to string, remove markdown bold, brackets, and strip
        s = str(value)
        s = re.sub(r'\*\*', '', s)       
        s = re.sub(r'[\[\]]', '', s)    
        s = s.strip().lower()


        if s == "not applicable":
          return "not applicable"

        # Explicit ambiguous / missing values
        if s in ["yes/no", "n/a", "", "none", "null"]:
            return "N/A"

        # Prefix checks
        if s.startswith("yes"):
            return "Yes"
        if s.startswith("no"):
            return "No"

        return "N/A"

    cleaned_series = df[column_name].apply(_clean_single)

    if new_column_name is None:
        df[column_name] = cleaned_series
    else:
        df[new_column_name] = cleaned_series

    return df



def parse_mixed_dates(df, column_name, new_column_name=None, inplace=False):
    """
    Parse a column containing dates in mixed formats into a unified datetime column.

    Handles three common patterns:
        - MM/DD/YYYY (e.g., 03/14/2025)
        - DD-MM-YYYY (e.g., 14-03-2025)
        - Any other format that pandas can infer (e.g., "2025-03-14")

    Invalid or missing values become `pd.NaT`.

    Args:
        df (pd.DataFrame): Input DataFrame.
        column_name (str): Name of the column containing date strings.
        new_column_name (str, optional): Name for the parsed datetime column.
            If None, the original column is replaced. Defaults to None.
        inplace (bool, optional): If True, modify the original DataFrame; otherwise,
            return a copy. Defaults to False.

    Returns:
        pd.DataFrame: DataFrame with the parsed datetime column (dtype datetime64[ns]).

    Examples:
        >>> import pandas as pd
        >>> df = pd.DataFrame({"date": ["03/14/2025", "14-03-2025", "2025-03-14", "invalid"]})
        >>> parse_mixed_dates(df, "date")
                         date
        0 2025-03-14 00:00:00
        1 2025-03-14 00:00:00
        2 2025-03-14 00:00:00
        3                  NaT
    """

    if not inplace:
        df = df.copy()

    def parse_single(date_str):
        if pd.isna(date_str) or date_str == "":
            return pd.NaT
        date_str = str(date_str).strip()
        if "/" in date_str:
            # MM/DD/YYYY format
            return pd.to_datetime(date_str, format="%m/%d/%Y", errors="coerce")
        elif "-" in date_str:
            # DD-MM-YYYY format
            return pd.to_datetime(date_str, format="%d-%m-%Y", errors="coerce")
        else:
            # fallback: let pandas try to infer
            return pd.to_datetime(date_str, errors="coerce")

    parsed = df[column_name].apply(parse_single)

    out_col = new_column_name if new_column_name is not None else column_name
    df[out_col] = parsed
    return df