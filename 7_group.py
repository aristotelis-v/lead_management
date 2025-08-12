import os
import pandas as pd
import numpy as np
import hashlib
import phonenumbers
import logging

from pathlib import Path
from functools import lru_cache
from datetime import datetime, timedelta

from tools.excel import build_custom_workbook

DATASOURCE_DIR = Path("datasources")
PARQUET_EXT     = ".parquet"

@lru_cache(maxsize=8)
def load_datasource(name: str, directory: Path = DATASOURCE_DIR, ext: str = PARQUET_EXT, engine: str = "pyarrow") -> pd.DataFrame:
    """
    Load a pandas DataFrame from a Parquet file, with caching and error handling.
    """
    path = directory / f"{name}{ext}"
    try:
        return pd.read_parquet(path, engine=engine)
    except FileNotFoundError:
        logging.error(f"Datasource not found: {path}")
        raise
    except Exception as e:
        logging.error(f"Failed to load parquet {path}: {e!r}")
        raise

def move_column_next_to(df, target_col, col_to_move):
    """
    Move an existing column to appear next to another column in a DataFrame.

    Parameters:
        df (pd.DataFrame): The input DataFrame.
        target_col (str): The name of the column to move next to.
        col_to_move (str): The name of the existing column to reposition.

    Returns:
        pd.DataFrame: A new DataFrame with columns reordered.
    """
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found.")
    if col_to_move not in df.columns:
        raise ValueError(f"Column to move '{col_to_move}' not found.")
    if target_col == col_to_move:
        return df.copy()

    cols = df.columns.tolist()
    cols.remove(col_to_move)
    target_index = cols.index(target_col)
    cols.insert(target_index + 1, col_to_move)

    return df[cols]

def replace_none(df, column):

    df[column] = df[column].replace('none', np.nan)

    return df

def calculate_flag(df, old_column, new_column):

    df[new_column] = (df[old_column].notna()).astype(int)

    df = move_column_next_to(df, old_column, new_column)

    return df

if __name__ == "__main__":
	
    df = load_datasource('blend')

    df = replace_none(df, 'initial_lead_status')
    df = replace_none(df, 'lead_status')

    df = calculate_flag(df, 'initial_lead_status', 'initial_lead_status_flag')
    df = calculate_flag(df, 'lead_status', 'lead_status_flag')

    print(df.head())

    # sheet_name_list = []

    # sheet_name_list.append(('source', df))

    cols = [
                'desc_web_site', 
                'cl_country_name', 
                'initial_platform', 
                'account_license', 
                'account_status', 
                'initial_lead_status', 
                'lead_status', 
                'age_group', 
                'annual_income', 
                'savings', 
                'knowledge_of_trading', 
                'os', 
                'sms_verification', 
                'demo_trade_flag', 
                'self_reg_real', 
                'dummy_db_flag', 
                'dummy'
            ]
    gf = df.groupby(cols, dropna=False).agg(
                                            ftds=('fx_ftd', 'sum'),
                                            ils_count=('initial_lead_status_flag', 'sum'),
                                            accounts=('account_id', 'count')
                                        ).reset_index()

    gf['l2ftd'] = gf['ftds'] / gf['accounts']
    gf['ils_l2ftd'] = gf['ftds'] / gf['ils_count']

    # gf = gf[(gf['desc_web_site'] == 'gcmforex') & (gf['initial_platform'] == 'fx') & (gf['account_license'] == 'fx') & (gf['dummy'] == 1)].reset_index(drop=True)

    # gf = gf[(gf['desc_web_site'] == 'fortrade.com') & (gf['dummy_db_flag'] == 1)]
    gf = gf[(gf['desc_web_site'] == 'fortrade.com') & (gf['dummy'] == 1)]

    print(gf.head())
    print(gf.shape)

    out_dir = 'datasources'
    os.makedirs(out_dir, exist_ok=True)
    gf.to_parquet(
        os.path.join(out_dir, 'group.parquet'),
        engine='pyarrow',
        compression='snappy'
    )

    # Excel

    # build_custom_workbook(sheet_name_list, f'dsb_analysis')
