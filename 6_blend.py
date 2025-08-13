import os
import pandas as pd
import numpy as np
import hashlib
import phonenumbers
import logging

from pathlib import Path
from functools import lru_cache
from datetime import datetime, timedelta
from user_agents import parse

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

def extract_user_agent_details(df: pd.DataFrame, ua_col: str = "lv_user_agent", inplace: bool = True) -> pd.DataFrame:
    """
    Parse user-agent strings from df[ua_col] and add columns:
      - browser, os, os_name, device, device_type

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame containing the user-agent column.
    ua_col : str, default "lv_user_agent"
        Name of the column with user-agent strings.
    inplace : bool, default True
        If True, mutate df. If False, return a new DataFrame with added columns.

    Returns
    -------
    pd.DataFrame
        The mutated df if inplace=True, otherwise a new DataFrame with added columns.
    """

    def _parse_one(ua_string):
        if pd.isna(ua_string) or not str(ua_string).strip():
            return {
                "browser": "",
                "os": "",
                "os_name": "",
                "device": "",
                "device_type": "Other"
            }
        try:
            ua = parse(str(ua_string))

            if ua.is_mobile:
                device_type = "Mobile"
            elif ua.is_tablet:
                device_type = "Tablet"
            elif ua.is_pc:
                device_type = "Desktop"
            else:
                device_type = "Other"

            return {
                "browser": f"{ua.browser.family} {ua.browser.version_string}".strip(),
                "os": f"{ua.os.family} {ua.os.version_string}".strip(),
                "os_name": ua.os.family or "",
                "device": f"{ua.device.family} (Brand: {ua.device.brand}, Model: {ua.device.model})",
                "device_type": device_type
            }
        except Exception:
            # Fallback on any parsing error
            return {
                "browser": "",
                "os": "",
                "os_name": "",
                "device": "",
                "device_type": "Other"
            }

    details = df[ua_col].apply(_parse_one).apply(pd.Series)

    if inplace:
        df[["browser", "os", "os_name", "device", "device_type"]] = details
        return df
    else:
        return df.join(details)

def has_valid_format(number, region=None):
    try:
        parsed = phonenumbers.parse(number, region)
        return phonenumbers.is_valid_number(parsed)
    except phonenumbers.NumberParseException:
        return False

def hash_to_upper(val):

    if pd.isnull(val):
        return None

    val_lower = str(val).lower()
    hash_bytes = hashlib.sha256(val_lower.encode('utf-8')).hexdigest()

    return hash_bytes.upper()

def process_vrfcs(vr: pd.DataFrame) -> pd.DataFrame:

    vr = vr.drop(columns=['conversion_status', 'mcc', 'mnc', 'code_length']).copy()

    vr['date_created'] = vr['date_created'].dt.tz_localize(None)
    vr['date_updated'] = vr['date_updated'].dt.tz_localize(None)

    idx = vr.columns.get_loc('message_status') + 1
    vr.insert(idx, 'message_status_code', vr['message_status'].map({'READ': 0, 'DELIVERED': 1, 'UNDELIVERED': 2, 'FAILED': 3, 'DELIVERY_UNKNOWN': 4, '': 5}))

    idx = vr.columns.get_loc('verification_status') + 1
    vr.insert(idx, 'verification_status_code', vr['verification_status'].map({'CONFIRMED': 0, 'UNCONFIRMED': 1}))

    vr['to'] = vr['to'].astype(str)
    vr['to_has_valid_format'] = vr['to'].apply(lambda x: has_valid_format(x))

    vr['to'] = vr['to'].str.replace(r'^.*?\+', '', regex=True)
    vr['to_hashed'] = vr['to'].apply(hash_to_upper)

    vr = vr.sort_values(by=['country', 'date_created'], ignore_index=True)

    return vr

def verify_sms(ld: pd.DataFrame, vr: pd.DataFrame) -> pd.DataFrame:

    ld = ld[['account_id', 'created_on', 'lv_fixed_phone1_gl', 'lv_fixed_phone2_gl', 'lv_fixed_phone1_fb', 'lv_fixed_phone2_fb', 'lc_sms_verification']].copy()
    vr = vr[['verification_attempt_sid', 'date_created', 'message_status', 'message_status_code', 'verification_status', 'verification_status_code', 'to_hashed']].copy()

    ld_melted = ld.melt(id_vars=['account_id'], value_vars=['lv_fixed_phone1_gl', 'lv_fixed_phone2_gl', 'lv_fixed_phone1_fb', 'lv_fixed_phone2_fb'], value_name='to_hashed').rename(columns={'variable': 'phone_label'}).dropna(subset=['to_hashed'])

    df = ld_melted.merge(vr, on='to_hashed', how='inner')
    df = ld.merge(df, on='account_id', how='left')

    df['attempts'] = df.groupby('account_id')['verification_attempt_sid'].transform('count')

    df['account_creation_delay'] = ((df['created_on']-df['date_created']).dt.total_seconds())
    df['correl_delay'] = df['account_creation_delay'].abs()

    # Pick the most relevant (i.e. closest to account creation) successful verification attempt per account. Note that multiple verification attempts for multiple numbers may exist.
    df = df.sort_values(['message_status_code', 'verification_status_code', 'correl_delay'], ascending=[True, True, True]).groupby('account_id', as_index=False).first()
    df = df.drop(columns=['lv_fixed_phone1_gl', 'lv_fixed_phone2_gl', 'lv_fixed_phone1_fb', 'lv_fixed_phone2_fb', 'phone_label', 'date_created', 'to_hashed', 'account_creation_delay', 'correl_delay'])

    df['sms_verification'] = np.where(
                                            df['lc_sms_verification'].isna() & df['verification_status'].isna(),
                                            np.nan,
                                            np.maximum(df['lc_sms_verification'].fillna(0).astype(int), (df['verification_status'] == 'CONFIRMED').astype(int))
                                        )

    df = df[['account_id', 'sms_verification']]

    return df

def extract_link_parts(df: pd.DataFrame, col: str = "link_id") -> pd.DataFrame:
    s = df[col].astype(str)

    # grab the first token in the comma-list that matches each category
    age_group = (
        s.str.findall(r'[^,]*age[^,]*', flags=0).str[0]
        .str.lower()
        .fillna('')
        .str.replace(r'less_than_18_age\w*', 'under_18_age', regex=True)
    )

    annual_income = (
        s.str.findall(r'[^,]*annual[^,]*').str[0]
        .str.lower()
        .fillna('')
    )

    savings = (
        s.str.findall(r'[^,]*savings[^,]*').str[0]
        .str.lower()
        .fillna('')
    )

    # knowledge_of_trading

    # s is df['link_id']
    tok = (
        s.str.extract(r'([^,]*knowledge_of_trading[^,]*)', expand=False)
         .fillna('')
         .str.strip()
         .str.lower()
    )

    # strip the prefix once
    tail = tok.str.replace(r'^\s*knowledge_of_trading_?', '', regex=True)

    # map many variants in one vectorized pass
    patterns = {
        r'^yes_a_r\w*$':        'financial_qualification',
        r'^yes_from_a_r\w*$':   'role_in_financial_services',
        r'^yes_from_p\w*$':     'previous_trading_experience',
        r'^all_the_above$':     'all_the_above',
        r'^fair$':              'fair',
        r'^good$':              'good',
        r'^limited$':           'limited',
        r'^none$':              'none',
    }

    knowledge_of_trading = tail.replace(patterns, regex=True)

    # OS: hyphenated with letters only OR known OS names
    os_ = (
        s.str.findall(r'\b[a-zA-Z]+-[a-zA-Z]+\b|macos|windows|linux').str[0]
        .str.lower()
        .fillna('')
        .str.replace(r'^pc-w\w*', 'pc-windows', regex=True)
        .str.replace(r'^mac(?:-[a-z]+)?$', 'mac-macos', regex=True)
        .str.replace(r'^(t|m)-a\w*', lambda m: f"{m.group(1)}-android", regex=True)
        .str.replace(r'^(t|m)-i\w*', lambda m: f"{m.group(1)}-ios", regex=True)
    )

    out = pd.DataFrame({
        'age_group': age_group,
        'annual_income': annual_income,
        'savings': savings,
        'knowledge_of_trading': knowledge_of_trading,
        'os_link_id': os_,
    })

    # return df.drop(columns=[col]).assign(**out.to_dict(orient='series'))
    return df.assign(**out.to_dict(orient='series'))

def select_proper_tp(df: pd.DataFrame) -> pd.DataFrame:

    df = df[['account_pk', 'tp_id', 'tp_pk', 'tp_created_on', 'tp_is_demo']]
    df = df.copy().sort_values(by=['account_pk', 'tp_is_demo', 'tp_created_on'])

    return df.groupby('account_pk', as_index=False).first()

def identify_self_registered(df: pd.DataFrame) -> pd.DataFrame:

    df['self_reg_real'] = (df['account_status'].str.contains('real', na=False) & (df['first_call_date'].isna() | (df['tp_created_on'] < df['first_call_date']))).astype(int)

    return df

def identify_dummy(df: pd.DataFrame) -> pd.DataFrame:

    df['dummy'] = np.where(
                            df['desc_web_site'] == 'gcmforex',
                            (df['detailed_tag'].str.endswith('@') & df['google_id'].str.endswith('@d')).astype(int),
                            df['age_group'].isin(['under_18_age', '18_24_age']).astype(int)
                        )
    return df

def count_demo_trades(df: pd.DataFrame) -> pd.DataFrame:

    df['demo_trade'] = 1 - df['real_trade']

    df['demo_trade_<_first_call'] = ((df['demo_trade'] == 1) & (df['first_call_date'].isna() | (df['open_time_utc'] < df['first_call_date']))).astype(int)

    gf = df.groupby('account_pk').agg(**{'demo_trades_<_first_call': ('demo_trade_<_first_call', 'sum')}).reset_index()

    gf['demo_trade_flag'] = (gf['demo_trades_<_first_call'] > 0).astype(int)

    gf = gf[['account_pk', 'demo_trade_flag']]

    return gf

def blend_datasources(db: pd.DataFrame, vr: pd.DataFrame, dw: pd.DataFrame, tp: pd.DataFrame, td: pd.DataFrame) -> pd.DataFrame:
    
    vr = process_vrfcs(vr)
    dr = verify_sms(db, vr)
    dw = extract_link_parts(dw, col="link_id")
    tp = select_proper_tp(tp)
    td = count_demo_trades(td)

    db = db.drop(columns=['created_on', 'lv_fixed_phone1_gl', 'lv_fixed_phone2_gl', 'lv_fixed_phone1_fb', 'lv_fixed_phone2_fb', 'lc_sms_verification'], errors='ignore')
    db = extract_user_agent_details(db)

    df = db.merge(dw, on='account_id', how='left')
    df = df.merge(dr, on='account_id', how='left')
    df = df.merge(tp, on='account_pk', how='left')
    df = df.merge(td, on='account_pk', how='left')

    df = identify_self_registered(df)
    df = identify_dummy(df)

    return df

if __name__ == "__main__":
	
    db = load_datasource('db')
    vr = load_datasource('vr')
    dw = load_datasource('dw')
    tp = load_datasource('tp')
    td = load_datasource('td')

    fc = dw[['account_pk', 'first_call_date']]

    td = td.merge(fc, on=['account_pk'], how='left')

    print(db.head())
    print(vr.head())
    print(dw.head())
    print(tp.head())
    print(td.head())

    df = blend_datasources(db, vr, dw, tp, td)

    print(df.head())
    print(df.shape)

    out_dir = 'datasources'
    os.makedirs(out_dir, exist_ok=True)
    df.to_parquet(
        os.path.join(out_dir, 'blend.parquet'),
        engine='pyarrow',
        compression='snappy'
    )
