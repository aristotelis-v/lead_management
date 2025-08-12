import os
import pyodbc
import pandas as pd

def load_data(start_date: str, end_date: str) -> pd.DataFrame:
    """
    Pulls account & fx transaction data from the 'gcm' and 'lc' marketing DBs
    for accounts created between start_date and end_date.
    """
    dfs = []
    dbs = ['gcm', 'lc']
    conn_str_template = (
        'DRIVER={{SQL Server}};'
        'SERVER=10.46.7.166;'
        'DATABASE=crm_{db}_marketing;'
        'UID={user};'
        'PWD={pwd};'
    )
    credentials = {
        'user': os.getenv('DB_USER', 'marketing_user'),
        'pwd': os.getenv('DB_PWD', 'Market!23'),
    }

    sql_base = """
    WITH account AS (
        SELECT
            accountid               AS account_id,
            createdon               AS created_on,
            lv_fixed_phone1_gl,
            lv_fixed_phone2_gl,
            lv_fixed_phone1_fb,
            lv_fixed_phone2_fb,
            lc_sms_verification,
            CASE 
                WHEN lv_first_co_ownerid IN (
                    CAST('6546523A-AC21-EB11-80E4-005056972925' AS uniqueidentifier), -- FT Google Camp User
                    CAST('7FB17431-8D9B-EB11-B81B-D96FDBC6BD2F' AS uniqueidentifier), -- KapitalRS Dummy user
                    CAST('3ABF7D6C-C683-EF11-B83C-9975671EC6C6' AS uniqueidentifier), -- Dummy Marketing - GCM
                    CAST('30169DDC-A1F5-EA11-80F4-005056972723' AS uniqueidentifier)  -- GCMForex Google Camp User
                ) THEN 1
                ELSE 0
            END                      AS dummy_db_flag,
            ?                       AS db
        FROM accountbase
        WHERE
            lv_siteid IN ('3F37DBE2-FCB2-E511-80C6-005056A44066', '2D826510-00B3-E511-80C8-005056A42D66', '4D9FA608-D8D7-E611-80E0-005056974CDF')
            AND createdon BETWEEN ? AND ?
            AND lv_contains_test = 0
            AND lv_created_in_office = 0
            AND (lv_accountstatus NOT IN (5,6) OR lv_accountstatus IS NULL)
            {gcm_filter_clause}
    )
    SELECT
        account_id,
        created_on,
        lv_fixed_phone1_gl,
        lv_fixed_phone2_gl,
        lv_fixed_phone1_fb,
        lv_fixed_phone2_fb,
        lc_sms_verification,
        dummy_db_flag,
        db
    FROM account
    """

    gcm_filter_clause = """
      AND (
        (lv_viop_account_status <> '772400005' OR lv_viop_account_status IS NULL)
        AND (lv_single_stocks_account_status <> '772400005' OR lv_single_stocks_account_status IS NULL)
        AND (lv_varant_account_status <> '772400005' OR lv_varant_account_status IS NULL)
      )
    """

    for db in dbs:
        conn_str = conn_str_template.format(db=db, **credentials)
        filter_clause = gcm_filter_clause if db == 'gcm' else ''
        sql = sql_base.format(gcm_filter_clause=filter_clause)

        with pyodbc.connect(conn_str) as conn:
            # read_sql_query takes care of cursor and DataFrame creation
            df_part = pd.read_sql_query(
                sql, conn,
                params=[db, start_date, end_date]
            )

        dfs.append(df_part)

    return pd.concat(dfs, ignore_index=True)


if __name__ == "__main__":

    df = load_data('2024-01-01', '2025-07-01')
    print(df.head())
    print(df.shape)

    out_dir = 'datasources'
    os.makedirs(out_dir, exist_ok=True)
    df.to_parquet(
        os.path.join(out_dir, 'db.parquet'),
        engine='pyarrow',
        compression='snappy'
    )
