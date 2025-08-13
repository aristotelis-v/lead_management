import os
import re
import pyodbc
import pandas as pd

def to_datetime(df, column):

    df[column] = pd.to_datetime(df[column], errors='coerce')
    df.loc[df[column].dt.normalize() == pd.Timestamp('1900-01-01'), column] = pd.NaT

    return df

def normalize_product_names(df, column):

    patterns = [
        (r'(?i)\b(fx|forex)\b', 'fx'),
        (r'(?i)\b(single stock(s)?|ss)\b', 'ss'),
        (r'(?i)\bviop\b', 'vp'),
        (r'(?i)\boption\b', 'op'),
        (r'(?i)\bvarant\b', 'vr')
    ]
    
    def replace_with_patterns(text):
        text = text.lower()
        for pattern, replacement in patterns:
            text = re.sub(pattern, replacement, text)
        return text
    
    df[column] = df[column].astype(str).apply(replace_with_patterns)

    return df

def lower_turkish_to_latin(df, column):

    # Mapping of Turkish characters to their Latin equivalents
    turkish_to_latin_map = {
        'ç': 'c', 'Ç': 'C',
        'ğ': 'g', 'Ğ': 'G',
        'ı': 'i', 'İ': 'I',
        'ö': 'o', 'Ö': 'O',
        'ş': 's', 'Ş': 'S',
        'ü': 'u', 'Ü': 'U'
    }

    def turkish_to_latin(text):

        # Replace Turkish characters with Latin equivalents
        return ''.join(turkish_to_latin_map.get(char, char) for char in text)

    df[column] = df[column].astype(str).apply(turkish_to_latin).str.lower()

    return df

def assign_age_group(df: pd.DataFrame, birthdate_col: str = "birthdate") -> pd.DataFrame:
    """
    Compute age-group labels into df['age_group_dw'] based on a birthdate column.

    Buckets:
      - age < 18            -> 'under_18_age'
      - 18–24               -> '18_24_age'
      - 25–34               -> '25_34_age'
      - 35–44               -> '35_44_age'
      - 45–54               -> '45_54_age'
      - 55–64               -> '55_64_age'
      - 65–74               -> '65_74_age'
      - 75+                 -> '75_plus_age'
      - invalid / future    -> ''
    """
    # Parse to datetime; invalid -> NaT
    bdate = pd.to_datetime(df[birthdate_col], errors="coerce")

    # Today's date (no time) for precise birthday comparison
    today = pd.Timestamp.today().normalize()

    # Full years of age
    before_birthday = (today.month < bdate.dt.month) | (
        (today.month == bdate.dt.month) & (today.day < bdate.dt.day)
    )
    age = (today.year - bdate.dt.year) - before_birthday.astype("Int64")

    # Start with blanks
    grp = pd.Series("", index=df.index, dtype="string")

    # Valid ages: not null and non-negative
    valid = age.notna() & (age >= 0)

    grp.loc[valid & (age < 18)] = "under_18_age"
    grp.loc[valid & age.between(18, 24, inclusive="both")] = "18_24_age"
    grp.loc[valid & age.between(25, 34, inclusive="both")] = "25_34_age"
    grp.loc[valid & age.between(35, 44, inclusive="both")] = "35_44_age"
    grp.loc[valid & age.between(45, 54, inclusive="both")] = "45_54_age"
    grp.loc[valid & age.between(55, 64, inclusive="both")] = "55_64_age"
    grp.loc[valid & age.between(65, 74, inclusive="both")] = "65_74_age"
    grp.loc[valid & (age >= 75)] = "75_plus_age"

    df["age_group_dw"] = grp

    return df

def calculate_ftd_flag(df, column):

    df['fx_ftd'] = df[column].notna().astype(int)

    return df

def load_data(start_date: str, end_date: str) -> pd.DataFrame:

    conn_str_template = (
        'DRIVER={{SQL Server}};'
        'SERVER=10.46.9.18;'
        'DATABASE=dwh;'
        'UID={user};'
        'PWD={pwd};'
    )

    credentials = {
        'user': os.getenv('DB_USER', 'marketing_user'),
        'pwd': os.getenv('DB_PWD', 'Market!23'),
    }

    sql = """
        SELECT
            accountid                   account_id,
            accountpk                   account_pk,
            createdon                   created_on,
            maintpid                    main_tp_id,
            tag,
            tag1                        detailed_tag,
            googleid                    google_id,
            linkid                      link_id,
            birthdate,
            descwebsite                 desc_web_site,
            initialplatform             initial_platform,
            accountlicense              account_license,
            accountstatusname           account_status,
            getnextfirstcalldate        first_call_date,
            initialleadstatus           initial_lead_status,
            leadstatus                  lead_status,
            real_verified_on,
            fxftdcreatedon              fx_ftd_created_on,
            fxftdapprovedon             fx_ftd_approved_on
        FROM
            dwhdimaccount
        WHERE
            createdon BETWEEN ? AND ?
            AND
            lower(descwebsite) IN ('fortrade.com', 'gcmforex', 'kapitalrs')
    """

    conn_str = conn_str_template.format(**credentials)

    with pyodbc.connect(conn_str) as conn:

        df = pd.read_sql_query(sql, conn, params=[start_date, end_date])

        df = to_datetime(df, 'created_on')
        df = to_datetime(df, 'real_verified_on')
        df = to_datetime(df, 'fx_ftd_created_on')
        df = to_datetime(df, 'fx_ftd_approved_on')

        df['tag'] = df['tag'].str.lower()

        df = normalize_product_names(df, 'initial_platform')   
        df = normalize_product_names(df, 'account_license')

        df = lower_turkish_to_latin(df, 'account_status')
        df = lower_turkish_to_latin(df, 'initial_lead_status')
        df = lower_turkish_to_latin(df, 'lead_status')
        
        df = calculate_ftd_flag(df, 'fx_ftd_approved_on')

        df = assign_age_group(df)

        df = df.sort_values(by='created_on', ignore_index=True)

    return df


if __name__ == "__main__":

    df = load_data('2024-01-01', '2025-07-01')
    print(df.head())
    print(df.shape)

    out_dir = 'datasources'
    os.makedirs(out_dir, exist_ok=True)
    df.to_parquet(
        os.path.join(out_dir, 'dw.parquet'),
        engine='pyarrow',
        compression='snappy'
    )
